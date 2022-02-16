# This file contains just the executable code from the notebook presented in the workshop
# If you're using a Terminal, you can copy and paste each lines as you go to follow along.
#
# If you're using VS Code (not the notebook view) you can use `Alt + Enter ` (Windows) or
# `Option + Enter` (Mac) to run one line at a time.

##
using Pkg
Pkg.activate(".")
Pkg.instantiate() # Only need the first time

@code_native 1 * 1

write("pythontest1.py", """
import time
start = time.time()
for i in range(0,1_000_000):
    i * i
end = time.time()
elapsed = (end - start)

instrs = 3.5e9 * elapsed/1_000_000

print(f"Est. instructions per Python multiply: {instrs}")
""")

# You may need to change this if you want to run these comparisons yourself
# pypath = "python"
pypath = raw"C:\Users\William\miniconda3\python.exe"
run(`$pypath pythontest1.py`);

a = 1
b = 3.0
c = 1//2

α = 3a + 2c

∑x = sum(3xi^2 for xi = 1:10)

f(x) = 2x^2 + 3x^3 + 6

f(2)

∇²(σₐ) = √3 + log(σₐ)

∇²(12.0)

# Output a string (like Python print)
println("The answer to life, the universe, and everything is")

# Quick show for debugging
d = 42
@show d;

# Nice logging messages
@info "The answer ... is" d
@warn "But what is the question?"

@show typeof(1)
@show typeof(1.0)
@show typeof("abc");

using Downloads
filename = Downloads.download("https://wttr.in/")

println(readuntil(filename, "┌"))

using Statistics
mean([1,2,3])

using Images
filename = Downloads.download("https://github.com/JuliaLang/julia-logo-graphics/blob/master/images/julia-logo-color.png?raw=true")

load(filename)

x = [1, 2, 3]

push!(x, 4)

q = [1,2,3,4]
@show q[1] q[2] q[3];

@show q[begin]
@show q[end]
@show q[end - 1]

@show q[begin:end÷2]

@show q[begin:end÷2]

A = [
    1 2 3
    4 5 6
    7 8 9
]

A^2

exp(A)

A * [1, 2, 3]

A .* [1, 2, 3]

A.^2

[1 2 3] .* [
             1
             2
             3 ]

sin.(A)

strs = [
    "A",
    "B",
    "C",
]
strs2 = ["a" "b"]
strs .* strs2

1:5

y = vcat(1:5, 6:8)

@show length(A)
@show size(A)
@show eachindex(A);

a = randn(10)
mask = -0.8 .< a .< 0.8

a[mask]

if 2 ∈ (1,2,3,4)
    println("This is an if statement")
end

if 2 in (1,2,3,4)
    println("This way is fine too")
end

for i in 1:10
    println(i^2)
end

for i in 1:9, j in 'A':'G'
    print(j,i," ")
    if j == 'G'
        println()
    end
end

using Base.Threads
nthreads()

@threads for i in 1:100
    sum(rand(1000,1000).^2)
end

struct MyPoint
    x::Float64
    y::Float64
    z::Float64
end

p1 = MyPoint(1,2,3)

p1.x

Base.:-(p1::MyPoint, p2::MyPoint) = MyPoint(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

p1 = MyPoint(1,2,3)
p2 = MyPoint(1,3,4)
p2 - p1

⨳(p1::MyPoint, p2::MyPoint) = p1.x / p2.y + p1.y / p2.z + p1.z / p2.x

p1 ⨳ p2

using Symbolics
@variables u v w

expr = exp(u)^w / w

simplify(expr)

using ForwardDiff

model(x, μ, A, σ) = A * exp(-(x-μ)^2/σ)

xdat = 0:0.5:3
dat = sin.(xdat)

meansquare(d1, d2) = sqrt(mean((d1 .- d2).^2))

fit((μ, A, σ)) = meansquare(model.(xdat, μ, A, σ), dat)

fit((1, 0.1, 2))


ForwardDiff.gradient(fit, [1, 0.1, 2])

ForwardDiff.hessian(fit, [1, 0.1, 2])

using Measurements
a = 2 ± 1
b = 4 ± 2

a * b

using DataFrames

df = DataFrame(
    "A" =>  1:10,
    "B" => 11:20
)

using CSV

hgca = CSV.read("HGCA_vEDR3.csv", DataFrame)

nearby = filter(hgca) do row
    row.parallax_gaia > 30 # About 40pc
end


sort!(nearby, [:chisq], rev=true)

# And let's pick the top  1000
beststars = nearby[1:1000, :]

# README!! Select one of the following:

# VSCode notebook or terminal:
# using WGLMakie

# Jupyter or VSCode terminal: (not 3D or interactive)
# using CairoMakie
# CairoMakie.activate!(type="svg")

# Terminal/desktop:
# using GLMakie

# CairoMakie: for nice PDF exports, figures for papers
# GLMakie   : for interactive or 3D plots in a separate window
# WGLMakie  : for quick interactive plots in Jupyter
if ! @isdefined Makie
    println("Error: please select one of the above plotting packaged before continuing! 👆")
end

fontsize_theme = Theme(fontsize =20)
set_theme!(fontsize_theme)

lines(1:180, sind.(1:180), axis=(xlabel="x", ylabel="y"))

x = range(-π, π, length=100)
y = range(-π, π, length=100)
z = sinc.(sqrt.(x.^2 .+ y'.^2))

surface(x, y, z, colormap=:plasma)

fig = Figure(
    resolution=(800,800)
)

xx = π/2*randn(1000)
yy = π/2*randn(1000)

ax1 = Makie.Axis(fig[1,1], xlabel="x", ylabel="y")
scatter!(ax1, xx, yy, )

ax2 = Makie.Axis(fig[2,1], xlabel="x", ylabel="y")
h = contourf!(ax2, x, y, z)

Colorbar(fig[1:2, 2], h, label="Colorbar")


linkxaxes!(ax1, ax2)

fig

fig,ax,pl = scatter(
    beststars.gaia_ra,
    beststars.gaia_dec,
    markersize=beststars.parallax_gaia ./ 5,
    color=log.(beststars.chisq),
    colormap=:turbo,
    axis=(;
        backgroundcolor=:black,
        gridcolor=:white,
        xlabel="RA",
        ylabel="DEC",
    )
)
Colorbar(fig[1,2], pl, label="significance")
fig

ρ = 1 ./ (nearby.parallax_gaia .* 1e-3)
φ = nearby.gaia_ra
θ = nearby.gaia_dec
rv = nearby.radial_velocity
rv[ismissing.(rv)] .= 0

# Spherical to Cartesian conversion
x = @. ρ * sin(φ) * cos(θ)
y = @. ρ * sin(φ) * sin(θ)
z = @. ρ * cos(φ)
# scatter(x, y, z)
fig, ax, pl = scatter(
    x,
    y,
    z,
    markersize=1000,
    color=atan.(rv),
    colormap = :bkr,
    axis = (;
        backgroundcolor=:black
    ),
    figure = (;
        resolution = (1200,900),
        backgroundcolor=:black

    )
)
scatter!(ax, [0],[0],[0], marker='⋆', color=:yellow, markersize=10000)
fig

N = 40
x = 20rand(N)
m = 0.4
b = 4
y = m .* x .+ b .+ 2randn(N)

scatter(
    x, y,
    axis=(
        xlabel="x",
        ylabel="y",
    )
)

using Turing

# Define a simple Normal model with unknown mean and variance.
@model function linear_regression_1(x, y)

    # Priors on slope, itercept, and variance
    m  ~ Normal(sqrt(10))
    σ₂ ~ TruncatedNormal(0, 100, 0, Inf)
    β  ~ Normal(0, sqrt(3))

    # Equation of a line
    μ = β .+ m .* x

    # We model our y points has being drawn from a Normal distribution about that line
    y ~ MvNormal(μ, sqrt(σ₂))
end

#  Run sampler, collect results
model = linear_regression_1(x, y)
chain = sample(model, NUTS(0.65), 3_000)

# Or run three chains in parallel using multiple threads:
# chain = sample(model, NUTS(0.65), MCMCThreads(), 3_000, 3)

# Make a traceplot
series(chain["m"]')

# We'll plot our posterior samples as lines with 200 steps between 0 and 20
xpost = range(-5, 25, length=200)

# Grab 300 posterior samples at random
ii = rand(eachindex(chain["m"]), 300)
ypost = chain["m"][ii] .* xpost' .+ chain["β"][ii]

fig, ax, pl = series(
    xpost,ypost,

    solid_color=(:black, 0.02),
    label="posterior",
    axis=(
        xlabel="x",
        ylabel="y",
    ),

)

# Overplot our data
s = scatter!(ax, x, y, label="data")

Legend(fig[1,2], [s,pl], ["data","posterior"], "Legend")

xlims!(ax, low=-2, high=22)

fig

hist(vec(chain["m"]), axis=(;xlabel="m"))

using PyCall

np = pyimport("numpy")

a = np.array(A)

@time np.mean(a)

@time mean(a)

py"""

print("Hello from Python!")

"""

using CUDA

# Create 10000 random floats
arr = randn(Float32, 10000)

# Transfer them to the GPU
cu_arr = CuArray(arr)

# Calculate the sum of squares now using your GPU! It's that easy!
sum(cu_arr .^2)
