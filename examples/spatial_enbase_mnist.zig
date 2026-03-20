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

fn imageEntropyWithWindow(
    image: []const u8,
    rows: usize,
    cols: usize,
    window: usize,
) !f64 {
    if (window == 0 or window % 2 == 0) return error.InvalidWindow;
    if (image.len != rows * cols) return error.InvalidImageSize;

    // ComputeLocalEntropy(image, W): local entropy for each pixel neighborhood.
    const radius: isize = @intCast(window / 2);
    const patch_area_f64 = @as(f64, @floatFromInt(window * window));

    var total_entropy: f64 = 0.0;

    var x: usize = 0;
    while (x < rows) : (x += 1) {
        var y: usize = 0;
        while (y < cols) : (y += 1) {
            var histogram = [_]u32{0} ** PixelBins;

            var dx: isize = -radius;
            while (dx <= radius) : (dx += 1) {
                const xx_i: isize = @as(isize, @intCast(x)) + dx;
                const xx_clamped = @max(@as(isize, 0), @min(@as(isize, @intCast(rows - 1)), xx_i));
                const xx: usize = @intCast(xx_clamped);

                var dy: isize = -radius;
                while (dy <= radius) : (dy += 1) {
                    const yy_i: isize = @as(isize, @intCast(y)) + dy;
                    const yy_clamped = @max(@as(isize, 0), @min(@as(isize, @intCast(cols - 1)), yy_i));
                    const yy: usize = @intCast(yy_clamped);

                    const px = image[xx * cols + yy];
                    histogram[px] += 1;
                }
            }

            var local_entropy: f64 = 0.0;
            for (histogram) |count| {
                if (count == 0) continue;
                const p = @as(f64, @floatFromInt(count)) / patch_area_f64;
                local_entropy -= p * std.math.log2(p);
            }

            total_entropy += local_entropy;
        }
    }

    // H_score <- Mean(E_map)
    return total_entropy / @as(f64, @floatFromInt(rows * cols));
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
    if (scratch.len % 2 == 1) return scratch[mid];
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

const ScorePair = struct {
    index: usize,
    score: f64,
};

fn compareScoreDesc(_: void, a: ScorePair, b: ScorePair) bool {
    return a.score > b.score;
}

fn spatialEnbaseSelectIndices(
    allocator: std.mem.Allocator,
    images: []const u8,
    labels: []const u8,
    rows: usize,
    cols: usize,
    window: usize,
) ![]usize {
    // X_selected / Y_selected are represented by selected global indices.
    var selected: std.ArrayList(usize) = .empty;
    errdefer selected.deinit(allocator);

    const image_size = rows * cols;

    var class_id: u8 = 0;
    while (class_id < ClassCount) : (class_id += 1) {
        // C <- indices belonging to class label.
        var class_indices = try collectClassIndices(allocator, labels, class_id);
        defer class_indices.deinit(allocator);

        if (class_indices.items.len == 0) continue;

        // M_Score <- {(index, H_score)} for each sample in C.
        var scored = try allocator.alloc(ScorePair, class_indices.items.len);
        defer allocator.free(scored);

        var i: usize = 0;
        while (i < class_indices.items.len) : (i += 1) {
            const global_idx = class_indices.items[i];
            const start = global_idx * image_size;
            const end = start + image_size;

            const score = try imageEntropyWithWindow(images[start..end], rows, cols, window);
            scored[i] = .{ .index = global_idx, .score = score };
        }

        // Sort M_Score in descending order by H_score.
        std.mem.sort(ScorePair, scored, {}, compareScoreDesc);

        var only_scores = try allocator.alloc(f64, scored.len);
        defer allocator.free(only_scores);

        for (scored, 0..) |pair, j| {
            only_scores[j] = pair.score;
        }

        // median <- median(scores in M_Score)
        const class_median = try median(allocator, only_scores);

        // IQualified <- {key.index | key.score >= median}
        for (scored) |pair| {
            if (pair.score >= class_median) {
                try selected.append(allocator, pair.index);
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

fn parseOptionalUsize(text: ?[]const u8, default_value: usize) !usize {
    if (text == null) return default_value;
    return std.fmt.parseInt(usize, text.?, 10);
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
    const window = try parseOptionalUsize(args.next(), 3);
    const max_samples = try parseOptionalUsize(args.next(), 5000);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = std.fs.File.stdout();
    var stdout_writer = stdout_file.writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.print("Spatial-EnBaSe for MNIST in Zig\n", .{});
    try stdout.print("Images IDX: {s}\n", .{images_path});
    try stdout.print("Labels IDX: {s}\n", .{labels_path});
    try stdout.print("Window W: {d}\n", .{window});
    try stdout.print("Max samples: {d}\n\n", .{max_samples});

    var mnist = loadMnist(allocator, images_path, labels_path) catch |err| {
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

    if (max_samples == 0) return error.InvalidMaxSamples;

    const limited_count = @min(mnist.count, max_samples);
    const limited_images = mnist.images[0 .. limited_count * mnist.rows * mnist.cols];
    const limited_labels = mnist.labels[0..limited_count];

    const selected = try spatialEnbaseSelectIndices(
        allocator,
        limited_images,
        limited_labels,
        mnist.rows,
        mnist.cols,
        window,
    );
    defer allocator.free(selected);

    const full_hist = classHistogram(limited_labels);
    const selected_hist = selectedClassHistogram(limited_labels, selected);

    try stdout.print("Total samples: {d}\n", .{limited_count});
    try stdout.print("Selected by Spatial-EnBaSe: {d}\n", .{selected.len});
    try stdout.print("Selection ratio: {d:.2}%\n\n", .{
        (@as(f64, @floatFromInt(selected.len)) / @as(f64, @floatFromInt(limited_count))) * 100.0,
    });

    try stdout.print("Class distribution (full -> selected):\n", .{});
    var label: usize = 0;
    while (label < ClassCount) : (label += 1) {
        try stdout.print("  {d}: {d} -> {d}\n", .{ label, full_hist[label], selected_hist[label] });
    }
}
