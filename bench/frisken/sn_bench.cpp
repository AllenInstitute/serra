// Benchmark Frisken's reference SurfaceNets implementation.
//
// The reference ships as a Visual Studio solution, but only Source/Application
// needs Qt and OpenGL. Source/SNLib is ISO C++ with nothing but standard
// library includes, so it builds with a plain clang++ and no port. This driver
// is the missing main(): it reads a raw uint16 volume, runs the three phases
// with timers, and prints one JSON line using the same keys as
// bench/compare_zmesh.py so the result drops straight into that table.
//
// Run one implementation per process. Peak RSS is the whole process, so a
// second mesher in the same address space would silently contaminate it.

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "MMCellFlag.h"
#include "MMGeometryGL.h"
#include "MMGeometryOBJ.h"
#include "MMSurfaceNet.h"

namespace {

// MMCellMap::Cell is private, so mirror its layout to report what the dense
// cell array costs. Declaring it here rather than hardcoding a byte count
// keeps the number honest when the patched build shrinks MMCellFlag.
struct CellLayout {
  unsigned short label;
  MMCellFlag flag;
  int vertexIndex;
  float vertexOffset[3];
};

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// macOS reports ru_maxrss in bytes, Linux in kibibytes. Decide from the
// platform, not from the magnitude: a value-based guess silently reports a
// 900 MB peak as 900 GB, which is exactly the range small volumes land in.
double peak_rss_gb() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  double raw = static_cast<double>(usage.ru_maxrss);
#ifdef __APPLE__
  return raw / 1e9;
#else
  return raw * 1024.0 / 1e9;
#endif
}

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "sn_bench: %s\n", message.c_str());
  std::exit(1);
}

// Parse "512,512,512" into three ints.
void parse_triple(const char *text, int out[3]) {
  if (std::sscanf(text, "%d,%d,%d", &out[0], &out[1], &out[2]) != 3) {
    fail(std::string("expected three comma-separated values, got: ") + text);
  }
}

void parse_triple(const char *text, float out[3]) {
  if (std::sscanf(text, "%f,%f,%f", &out[0], &out[1], &out[2]) != 3) {
    fail(std::string("expected three comma-separated values, got: ") + text);
  }
}

std::vector<unsigned short> read_volume(const std::string &path, size_t count) {
  std::FILE *handle = std::fopen(path.c_str(), "rb");
  if (!handle) fail("cannot open " + path);
  std::vector<unsigned short> data(count);
  size_t got = std::fread(data.data(), sizeof(unsigned short), count, handle);
  std::fclose(handle);
  if (got != count) {
    fail("short read from " + path + ": wanted " + std::to_string(count) +
         " voxels, got " + std::to_string(got));
  }
  return data;
}

struct Options {
  std::string volume;
  int dims[3] = {0, 0, 0};
  float voxel[3] = {1.0f, 1.0f, 1.0f};
  int relax = 0;
  float relax_factor = 0.5f;
  float max_dist = 1.0f;
  // How many labels to extract. Negative means all of them; objData() rescans
  // every quad once per label, so on a large volume "all" is minutes and a
  // sample is how you get a per-label cost without waiting for it.
  int sample = -1;
  bool skip_extract = false;
  bool gl = false;
  std::string dump_obj;
  int dump_label = -1;
};

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next = [&](const char *what) -> const char * {
      if (i + 1 >= argc) fail(std::string("missing value for ") + what);
      return argv[++i];
    };
    if (arg == "--volume") {
      opts.volume = next("--volume");
    } else if (arg == "--dims") {
      parse_triple(next("--dims"), opts.dims);
    } else if (arg == "--voxel") {
      parse_triple(next("--voxel"), opts.voxel);
    } else if (arg == "--relax") {
      opts.relax = std::atoi(next("--relax"));
    } else if (arg == "--relax-factor") {
      opts.relax_factor = static_cast<float>(std::atof(next("--relax-factor")));
    } else if (arg == "--max-dist") {
      opts.max_dist = static_cast<float>(std::atof(next("--max-dist")));
    } else if (arg == "--sample") {
      opts.sample = std::atoi(next("--sample"));
    } else if (arg == "--no-extract") {
      opts.skip_extract = true;
    } else if (arg == "--gl") {
      opts.gl = true;
    } else if (arg == "--dump-obj") {
      opts.dump_obj = next("--dump-obj");
    } else if (arg == "--dump-label") {
      opts.dump_label = std::atoi(next("--dump-label"));
    } else {
      fail("unknown argument: " + arg);
    }
  }
  if (opts.volume.empty()) fail("--volume is required");
  if (opts.dims[0] <= 0 || opts.dims[1] <= 0 || opts.dims[2] <= 0) {
    fail("--dims is required and must be positive");
  }
  return opts;
}

void write_obj(const std::string &path, const MMGeometryOBJ::OBJData &data,
               const float voxel[3]) {
  std::FILE *out = std::fopen(path.c_str(), "w");
  if (!out) fail("cannot write " + path);
  // Two corrections put these coordinates in the input volume's own frame.
  // Axes are reversed back into numpy/serra order (see the note in main()),
  // and one voxel is subtracted per axis: MMCellMap pads the array by a voxel
  // on every face and reports positions in the padded index space, so an
  // uncorrected mesh sits a voxel away from the segmentation it came from.
  for (const auto &p : data.vertexPositions) {
    std::fprintf(out, "v %.6f %.6f %.6f\n", p[2] - voxel[0], p[1] - voxel[1],
                 p[0] - voxel[2]);
  }
  // Reversing the axes is a reflection, so it flips the sign of every triangle.
  // Swapping two corners puts the outward normals back where they belong;
  // without it the mesh is inside out and its enclosed volume comes out
  // negative.
  for (const auto &t : data.triangles) {
    std::fprintf(out, "f %d %d %d\n", t[0], t[2], t[1]);
  }
  std::fclose(out);
}

}  // namespace

int main(int argc, char **argv) {
  Options opts = parse_args(argc, argv);

  size_t voxels = static_cast<size_t>(opts.dims[0]) * opts.dims[1] * opts.dims[2];
  std::vector<unsigned short> labels = read_volume(opts.volume, voxels);

  // --dims and --voxel are given in numpy/serra order, so axis 0 is the
  // slowest-varying one. MMCellMap walks i fastest, so its arraySize[0] is our
  // axis 2 and the two conventions are mirror images of each other. Reversing
  // here, and reversing again on OBJ output, means every coordinate this
  // program prints is directly comparable with a serra vertex -- rather than
  // leaving a silent transpose for the caller to discover on a non-cubic
  // volume, where it is a wrong answer instead of a crash.
  int array_size[3] = {opts.dims[2], opts.dims[1], opts.dims[0]};
  float voxel_size[3] = {opts.voxel[2], opts.voxel[1], opts.voxel[0]};

  // MMCellMap pads every face by one voxel with a reserved label, so the cell
  // array it allocates is (n+2)^3 and the padding closes the surface at the
  // volume boundary. Report the size it will really allocate rather than the
  // one we asked for -- that product times sizeof(Cell) is the memory story.
  double padded_cells = (static_cast<double>(opts.dims[0]) + 2) *
                        (static_cast<double>(opts.dims[1]) + 2) *
                        (static_cast<double>(opts.dims[2]) + 2);

  auto start = std::chrono::steady_clock::now();
  MMSurfaceNet net(labels.data(), array_size, voxel_size);
  double construct_s = seconds_since(start);
  double construct_peak_gb = peak_rss_gb();

  double relax_s = 0.0;
  if (opts.relax > 0) {
    MMSurfaceNet::RelaxAttrs attrs;
    attrs.numRelaxIterations = opts.relax;
    attrs.relaxFactor = opts.relax_factor;
    attrs.maxDistFromCellCenter = opts.max_dist;
    start = std::chrono::steady_clock::now();
    net.relax(attrs);
    relax_s = seconds_since(start);
  }

  // net.labels() is itself a full sweep over every vertex, so it is timed
  // separately rather than folded into extraction.
  start = std::chrono::steady_clock::now();
  std::vector<int> ids = net.labels();
  double labels_s = seconds_since(start);

  double quads_s = 0.0;
  double extract_s = 0.0;
  long long total_vertices = 0;
  long long total_triangles = 0;
  int extracted = 0;

  if (!opts.skip_extract) {
    // Building the quad list is a one-off shared by every label; objData() is
    // the per-label part. Timing them apart is the difference between "this
    // library is slow" and "this library rescans a global array per object".
    start = std::chrono::steady_clock::now();
    MMGeometryOBJ geometry(&net);
    quads_s = seconds_since(start);

    size_t limit = ids.size();
    if (opts.sample >= 0 && static_cast<size_t>(opts.sample) < limit) {
      limit = static_cast<size_t>(opts.sample);
    }
    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < limit; i++) {
      MMGeometryOBJ::OBJData data = geometry.objData(ids[i]);
      total_vertices += static_cast<long long>(data.vertexPositions.size());
      total_triangles += static_cast<long long>(data.triangles.size());
      extracted++;
      if (!opts.dump_obj.empty() && ids[i] == opts.dump_label) {
        write_obj(opts.dump_obj, data, opts.voxel);
      }
    }
    extract_s = seconds_since(start);
  }

  double gl_s = 0.0;
  long long gl_vertices = 0;
  long long gl_indices = 0;
  if (opts.gl) {
    // The whole net in one pass, for contrast with the per-label loop above.
    // It emits an unshared vertex per quad corner, so it is a soup.
    start = std::chrono::steady_clock::now();
    MMGeometryGL geometry(&net);
    gl_s = seconds_since(start);
    gl_vertices = geometry.numVertices();
    gl_indices = geometry.numIndices();
  }

  // Match compare_zmesh.py's accounting: float32 xyz per vertex, int32 per
  // triangle corner.
  double output_gb =
      (total_vertices * 3.0 * 4.0 + total_triangles * 3.0 * 4.0) / 1e9;

  std::printf(
      "{\"implementation\": \"frisken\", \"objects\": %zu, \"march_s\": %.4f, "
      "\"march_peak_gb\": %.3f, \"labels_s\": %.4f, \"relax_s\": %.4f, "
      "\"relax_iterations\": %d, \"quads_s\": %.4f, \"get_all_s\": %.4f, "
      "\"labels_extracted\": %d, \"vertices\": %lld, \"faces\": %lld, "
      "\"output_gb\": %.4f, \"peak_gb\": %.3f, \"padded_cells\": %.0f, "
      "\"cell_bytes\": %zu, \"cell_array_gb\": %.3f, \"gl_s\": %.4f, "
      "\"gl_vertices\": %lld, "
      "\"gl_indices\": %lld}\n",
      ids.size(), construct_s, construct_peak_gb, labels_s, relax_s,
      opts.relax, quads_s, extract_s, extracted, total_vertices,
      total_triangles, output_gb, peak_rss_gb(), padded_cells,
      sizeof(CellLayout), padded_cells * sizeof(CellLayout) / 1e9, gl_s,
      gl_vertices, gl_indices);
  return 0;
}
