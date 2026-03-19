const std = @import("std");

const ClassCount = 10;
const PixelBins = 256;

const MnistData = struct {
    images: []u8,
    labels: []u8,
    count: usize,
    rows: usize,
    cols: usize,

    fn deinit(self: MnistData, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
        allocator.free(self.labels);
    }
};

fn parseU32BE(bytes: []const u8) !u32 {
    if (bytes.len < 4) return error.InvalidHeader;
    return std.mem.readInt(u32, bytes[0..4], .big);
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const cwd = std.fs.cwd();
    return cwd.readFileAlloc(allocator, path, std.math.maxInt(usize));
}

fn loadMnist(allocator: std.mem.Allocator, images_path: []const u8, labels_path: []const u8) !MnistData {
    // Read raw IDX files to keep dependencies minimal and reproducible.
    const image_file = try readFileAlloc(allocator, images_path);
    errdefer allocator.free(image_file);

    const label_file = try readFileAlloc(allocator, labels_path);
    errdefer allocator.free(label_file);

    if (image_file.len < 16 or label_file.len < 8) return error.InvalidHeader;

    const image_magic = try parseU32BE(image_file[0..4]);
    const image_count_u32 = try parseU32BE(image_file[4..8]);
    const rows_u32 = try parseU32BE(image_file[8..12]);
    const cols_u32 = try parseU32BE(image_file[12..16]);

    const label_magic = try parseU32BE(label_file[0..4]);
    const label_count_u32 = try parseU32BE(label_file[4..8]);

    if (image_magic != 2051 or label_magic != 2049) return error.InvalidMagicNumber;

    const image_count = @as(usize, image_count_u32);
    const label_count = @as(usize, label_count_u32);
    const rows = @as(usize, rows_u32);
    const cols = @as(usize, cols_u32);
    const image_size = rows * cols;

    if (image_count != label_count) return error.CountMismatch;
    if (image_file.len != 16 + image_count * image_size) return error.InvalidImageLength;
    if (label_file.len != 8 + label_count) return error.InvalidLabelLength;

    const images = try allocator.alloc(u8, image_count * image_size);
    errdefer allocator.free(images);
    @memcpy(images, image_file[16..]);

    const labels = try allocator.alloc(u8, label_count);
    errdefer allocator.free(labels);
    @memcpy(labels, label_file[8..]);

    allocator.free(image_file);
    allocator.free(label_file);

    return MnistData{
        .images = images,
        .labels = labels,
        .count = image_count,
        .rows = rows,
        .cols = cols,
    };
}

fn imageEntropy(image: []const u8) f64 {
    // Entropy H = -sum(p * log2(p)) over the 256 gray-level histogram.
    var histogram = [_]u32{0} ** PixelBins;
    for (image) |px| {
        histogram[px] += 1;
    }

    const total = @as(f64, @floatFromInt(image.len));
    var entropy: f64 = 0.0;
    for (histogram) |count| {
        if (count == 0) continue;
        const p = @as(f64, @floatFromInt(count)) / total;
        entropy -= p * std.math.log2(p);
    }
    return entropy;
}

fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
    if (values.len == 0) return error.EmptyValues;

    const scratch = try allocator.dupe(f64, values);
    defer allocator.free(scratch);

    std.mem.sort(f64, scratch, {}, struct {
        fn lessThan(_: void, a: f64, b: f64) bool {
            return a < b;
        }
    }.lessThan);

    const mid = scratch.len / 2;
    if (scratch.len % 2 == 1) {
        return scratch[mid];
    }
    return (scratch[mid - 1] + scratch[mid]) / 2.0;
}

fn collectClassIndices(
    allocator: std.mem.Allocator,
    labels: []const u8,
    target: u8,
) !std.ArrayList(usize) {
    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);

    for (labels, 0..) |label, idx| {
        if (label == target) {
            try indices.append(allocator, idx);
        }
    }
    return indices;
}

fn enbaseSelectIndices(
    allocator: std.mem.Allocator,
    images: []const u8,
    labels: []const u8,
    image_size: usize,
) ![]usize {
    // Direct implementation of EnBaSe (returns selected global indices).
    var selected: std.ArrayList(usize) = .empty;
    errdefer selected.deinit(allocator);

    var class_id: u8 = 0;
    while (class_id < ClassCount) : (class_id += 1) {
        // C <- indices belonging to the current class.
        var class_indices = try collectClassIndices(allocator, labels, class_id);
        defer class_indices.deinit(allocator);

        if (class_indices.items.len == 0) continue;

        var entropies = try allocator.alloc(f64, class_indices.items.len);
        defer allocator.free(entropies);

        // MEntropy <- entropy map for the current class.
        for (class_indices.items, 0..) |global_idx, i| {
            const start = global_idx * image_size;
            const end = start + image_size;
            entropies[i] = imageEntropy(images[start..end]);
        }

        // median <- class entropy median.
        const class_median = try median(allocator, entropies);

        // IQualified <- samples with entropy <= median.
        for (class_indices.items, 0..) |global_idx, i| {
            if (entropies[i] <= class_median) {
                try selected.append(allocator, global_idx);
            }
        }
    }

    return selected.toOwnedSlice(allocator);
}

fn classHistogram(labels: []const u8) [ClassCount]usize {
    var counts = [_]usize{0} ** ClassCount;
    for (labels) |label| {
        if (label < ClassCount) counts[label] += 1;
    }
    return counts;
}

fn selectedClassHistogram(
    labels: []const u8,
    selected_indices: []const usize,
) [ClassCount]usize {
    var counts = [_]usize{0} ** ClassCount;
    for (selected_indices) |idx| {
        const label = labels[idx];
        if (label < ClassCount) counts[label] += 1;
    }
    return counts;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next();

    const default_images = "examples/data/train-images-idx3-ubyte";
    const default_labels = "examples/data/train-labels-idx1-ubyte";

    const images_path = args.next() orelse default_images;
    const labels_path = args.next() orelse default_labels;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = std.fs.File.stdout();
    var stdout_writer = stdout_file.writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("EnBaSe for MNIST in Zig\n", .{});
    try stdout.print("Images IDX: {s}\n", .{images_path});
    try stdout.print("Labels IDX: {s}\n\n", .{labels_path});

    const mnist = loadMnist(allocator, images_path, labels_path) catch |err| {
        try stdout.print("Failed to load MNIST: {s}\n", .{@errorName(err)});
        try stdout.print(
            "Download files and decompress to examples/data:\n" ++
                "  https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz\n" ++
                "  https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz\n",
            .{},
        );
        return err;
    };
    defer mnist.deinit(allocator);

    // Apply EnBaSe on the training set and report selected subset statistics.
    const image_size = mnist.rows * mnist.cols;
    const selected = try enbaseSelectIndices(allocator, mnist.images, mnist.labels, image_size);
    defer allocator.free(selected);

    const full_hist = classHistogram(mnist.labels);
    const selected_hist = selectedClassHistogram(mnist.labels, selected);

    try stdout.print("Total samples: {d}\n", .{mnist.count});
    try stdout.print("Selected by EnBaSe: {d}\n", .{selected.len});
    try stdout.print("Selection ratio: {d:.2}%\n\n", .{
        (@as(f64, @floatFromInt(selected.len)) / @as(f64, @floatFromInt(mnist.count))) * 100.0,
    });

    try stdout.print("Class distribution (full -> selected):\n", .{});
    var label: usize = 0;
    while (label < ClassCount) : (label += 1) {
        try stdout.print("  {d}: {d} -> {d}\n", .{ label, full_hist[label], selected_hist[label] });
    }
}
