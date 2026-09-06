# Does Circuitscape treat repeated labels in the point raster as one short-circuited focal region?
# Run: julia --project=. examples/region_check.jl <outdir>
using Circuitscape, DelimitedFiles
outdir = ARGS[1]; mkpath(outdir)
function write_asc(path, a)
    open(path, "w") do io
        println(io, "ncols $(size(a,2))\nnrows $(size(a,1))\nxllcorner 0\nyllcorner 0\ncellsize 100\nNODATA_value -9999")
        writedlm(io, a, ' ')
    end
end
n = 12
R = ones(n, n); R[:, 6] .= 50.0                  # a costly column in the middle
write_asc(joinpath(outdir, "res.asc"), R)
# A: label 1 on a whole column strip (12 px), label 2 on the opposite strip
P = zeros(Int, n, n); P[:, 1] .= 1; P[:, n] .= 2
write_asc(joinpath(outdir, "pts_strips.asc"), P)
# B: single pixels at the strip midpoints
Q = zeros(Int, n, n); Q[6, 1] = 1; Q[6, n] = 2
write_asc(joinpath(outdir, "pts_points.asc"), Q)
function run(pts, name)
    cfg = Dict{String,String}(
        "data_type" => "raster", "scenario" => "pairwise", "habitat_file" => joinpath(outdir, "res.asc"),
        "habitat_map_is_resistances" => "True", "point_file" => pts, "solver" => "cholmod",
        "connect_four_neighbors_only" => "False", "connect_using_avg_resistances" => "False",
        "write_cur_maps" => "True", "write_volt_maps" => "True", "output_file" => joinpath(outdir, name * ".out"),
        "suppress_messages" => "True", "log_level" => "INFO")
    Circuitscape.compute(cfg)
end
r_strips = run(joinpath(outdir, "pts_strips.asc"), "strips")
r_points = run(joinpath(outdir, "pts_points.asc"), "points")
println("resistances (strips):"); show(stdout, "text/plain", r_strips); println()
println("resistances (points):"); show(stdout, "text/plain", r_points); println()
cur = readdlm(joinpath(outdir, "strips_cum_curmap.asc"); skipstart = 6)
println("strip run: current in column 1 (should be spread over 12 pixels, not concentrated at one): ", round.(cur[:, 1]; digits = 3))
println("strip run: sum of current entering column 2 ≈ 1? ", round(sum(cur[:, 2]) , digits = 3))
println("REGION CHECK OK: strips Reff < points Reff -> ", r_strips[2, 3] < r_points[2, 3])
