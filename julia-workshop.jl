### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ c9be86ac-e5a8-46d5-8619-8bb39c236416
using BenchmarkTools

# ╔═╡ 00b6f7ef-6f55-40f4-a14f-1cfbd41215ac
using Downloads

# ╔═╡ 90a32f1e-27e0-489f-bf88-b029926ef0ee
using Statistics

# ╔═╡ 76bf1286-fcd1-4969-9c51-d7b8b4bcc138
using AstroImages

# ╔═╡ 71ea979a-2ce8-490b-886a-686b3917d34a
using ForwardDiff

# ╔═╡ 03378a27-5375-4359-b48c-42359f2d113d
using Measurements

# ╔═╡ 2bc67983-9907-47a1-9695-8970726b957f
using DataFrames

# ╔═╡ 3fd8a8b1-adbc-4bc0-bf64-6c668c35a67c
using CSV

# ╔═╡ 10292915-20b0-4838-a77e-258249b025b8
using WGLMakie; WGLMakie.activate!()

# ╔═╡ 548c1c6f-a752-4af9-8724-63ad8b3473ad
using Turing

# ╔═╡ 69998f08-1202-4220-a539-089fc947f80f
 using PairPlots

# ╔═╡ 02206a11-b3fe-4fb7-b0a0-5243527d744f
begin
	ENV["JULIA_CONDAPKG_BACKEND"] = "base"
	using PythonCall
end

# ╔═╡ 8424af6a-01b2-11ef-3775-6b696ebd1762
md"""
# Intro to Julia

William Thompson --- 2024-04-25



## Topics

- Speed
- Syntax
- Autodiff
- Data
- Plotting
- Modelling with Turing
- Using python packages

I'm going to be talking up Julia a lot during this workshop, but I want to note:

These are my own opinions
there's no universal "best" solution for everyone
I am aware that it's possible to do most things in other languages
Instead, I hope to show you some of the areas that Julia really shines.

I'm going to be contrasting Julia with Python a lot today since it's the language this audience is the most familiar with
But I love Python! I used it for years, it's great at what it does, and has massive library of thoughfully developed packages
"""
	

# ╔═╡ 70e030c4-592a-4df2-aa65-9b09c9808408
md"""
## Where we were

* Our goal should be to "do" science as efficiently and correctly as we can
* We want easy to reproduce results: free, open source, good package management
pip, conda, mamba, poetry, pyenv, ...?
* How many times have you got stuck for hours, or days trying to install a pacakge?
> "It works for my collaborator, why not me?"

> "I just deleted Python and installed Anaconda again for the fourth time"

* Code should be easy to read, write, and understand (no black boxes!)
* People want to develop their ideas in a convenient language like MATLAB or Python, but when their problems grows they need to stop and re-write it from scratch in a "hard" but fast language like C or Fortran. This is called the Two Language Problem.

This is a real impediment to research!

-----

Since Python is so slow, most numerical libraries are actually written in C or Fortran This leads to issues even if you never write your own pacakge:

* The pacakge you need might not compile on your system
* If you're using a library function, say numpy.median and you want to make your own version that's a little different, you can't (without learning all about C, compilers, combining C and Python, etc)
* Libraries don't combine easily (numpy vs. JAX vs. Torch vs. tensorflow)
* Numbers work differently depending on if they're in a NumPy array
* Lists vs arrays vs matrices?


Enter Julia....

---

## Why Julia

* Julia is a "new" programming language. 12 years old!

* Roughly a few 100,000 users

* Created at MIT by Alan Edelman and his graduate students, now co-founders of Julia Computing

* Free open source language but you can pay for support

* Designed from the start for science and numerical work

* Great package manager built in that just about guarantees reproducible environments

* Finally, it's very, very fast

Julia is an interative language just like Python. You can use it in in a terminal, in Jupyter notebooks, in Pluto Notebooks , VS Code, etc. But it's also a compiled language like C. The first time you run a function, it gets compiled to fast native code ("assembly").
"""

# ╔═╡ 4aaaed7d-7e6d-43a4-acf0-77bd0e7cb088
@code_native  1 * 1    # mul	x0, x1, x0

# ╔═╡ 708dea70-8e9e-4adb-bbfc-b19acbf22394
@btime Base.inferencebarrier(1 * 1);

# ╔═╡ 7454ba00-5671-42ac-aac5-94b6ea75b4eb
md"""
Most of this is function call overhead. There is really only one instruction executed, imulq. This is not the case with Python, MATLAB, R, etc. They read the text of your program and step through line by line.

We can estimate how many instructions Python uses to multiply two integers:
"""

# ╔═╡ fb9dc1d0-f277-4c38-a289-603b5f53cbd3
write("pythontest1.py", """
import time
start = time.time()
for i in range(0,1_000_000):
    i * i
end = time.time()
elapsed = (end - start)

instrs = 3.49e9 * elapsed/1_000_000

print(f"Est. instructions per Python multiply: {instrs}. Time: {elapsed*1000:.0f}ms")
""");

# ╔═╡ 42ea29e3-b48e-4f12-bf40-f24fd2407bd5
run(`python pythontest1.py`);

# ╔═╡ f820fee7-dea8-4329-8bae-ddbae130e802
md"""
This is a good estimate of how much faster Julia is than Python. When I run the same test using large NumPy arrays (best case) I get around 12 instructions.

In general, Julia is roughly a few hundred times faster than Python, and 10-50× faster than NumPy. For some applications like differential equations or optimization, Julia can be 350× faster than SciPy.
"""

# ╔═╡ cf265d34-740d-4e14-8bdd-2db70d9e5966
md"""
## So What?

Because Julia is so fast, you can solve problems on your own laptop that would otherwise need a compute cluster

Because it is compiled, you can write programs in natural ways without reaching for arrays all the time

You can see how Julia packages work and make your own changes. Black boxes are bad for science!

Let's dive in
"""

# ╔═╡ 6fa7c392-bfbb-4bb1-958c-693e08a95266
md"""
## Getting started

* Install Julia via `JuliaUp`: [julialang.org/downloads/](https://julialang.org/downloads/)

* Start Julia: run command `julia --threads=auto`

* Start pluto notebooks: `using Pluto; Pluto.run()`

* Paste URL to this notebook: `https://github.com/sefffal/JuliaNotes/blob/master/notes-1.ipynb`
"""

# ╔═╡ 18781b6c-7a89-44de-8c8b-7f72d8bc9b47
md"""
## Syntax

The Julia syntax should be quite familiar if you have used numpy, or MATLAB.
"""

# ╔═╡ 2401c963-0bba-42ae-adb8-39a5cd455ad9
md"
Variables:
"

# ╔═╡ 4d5351d9-c208-4c02-88f2-2d962e23f03a
a = 1

# ╔═╡ 578b0480-53d3-4c38-bff6-4d68fa341f7c
b = 3.0

# ╔═╡ e7ab3818-b6b7-46bd-a10f-5c0f1e079364
c = 1//2

# ╔═╡ 75505abe-c4ad-48c4-ae09-7f07f133e8a1
md"""
Mathematical Expressions
"""

# ╔═╡ d9e838ed-f5b2-4673-a2c5-a7be6b9e0793
α = 3a + 2c

# ╔═╡ 98651864-d363-4ad0-ab11-323bf81b9031
∑x = sum(3xi^2 for xi = 1:10)

# ╔═╡ 4837a067-da0d-4e99-9998-5ece7258fd10
md"Functions"

# ╔═╡ 77559e19-c77b-4f2b-9108-e1991c1d31a7
f(x) = 2x^2 + 3x^3 + 9

# ╔═╡ 0a48c9d6-d6d6-4c7c-96de-3f6c4bfe64c6
f(3)

# ╔═╡ b99279cb-65cb-4a9f-82a7-b5cac6f06849
md"""
## Mathemtical notation using $\LaTeX$

In Julia, you can type most basic LaTeX commands and hit `<Tab>`. Julia will autocomplete them and convert to Unicode.

For example, `\alpha` + `<Tab>` -> `α`.

There's a great font called JuliaMono that includes glyphs for almost the entire Unicode catalog. If you want your Julia code to look extra nice, give it a shot!
"""

# ╔═╡ b67ff7c5-674b-4b26-aee8-dd00b37526ca
∇²(σₐ) = √3 + log(σₐ)

# ╔═╡ 057f1d04-6629-4059-ab41-4def9e5dd565
∇²(12.0)

# ╔═╡ fae56d7e-de38-4d8a-bc75-7cb57d9b1467
md"The last value in a cell will be displayed automatically. You can also print things out directly using `print` and `println`"

# ╔═╡ 5aebc02a-85e0-43e5-a136-77c7abbdabba
print("Hello world")

# ╔═╡ 15322d81-9104-47b0-b65e-ef0f4ff078bd
md"The `@show` macro is a handy way to print out variables in your calculations."

# ╔═╡ 6a847a64-d3a1-499f-aae4-7754fb5cae63
@show a b c

# ╔═╡ 17871d0b-d248-43b8-aa0e-91e931afdee1
md"""
## Types

In Julia, every value has a type.
"""

# ╔═╡ 1852fa12-3af6-454f-861a-60168fe33581
typeof(1)

# ╔═╡ 66ee9d25-d1b3-4cd2-9f80-f9821d558ad6
typeof(1.0)

# ╔═╡ a5af4194-e39e-42c0-ac80-a0e592dd348c
typeof("abc")

# ╔═╡ 29d53a10-737a-4311-8d46-6b92ef058a51
md"""
Julia "infers" the types of your code. You rarely have to think about it unless working on a package.


!!! note
	Adding types does not speed up your code!
"""

# ╔═╡ b1db2c82-f835-4089-8b95-5b9a31ee42ec
md"""
## Importing Libraries

The normal way to import Julia libraries is with `using SomePackage`

This loads the package (if installed).

You can then access functions from that package like `SomePackage.func()`:
"""

# ╔═╡ 08afdbd6-20cd-402e-b21d-cff682287fd6
fname1 = Downloads.download("https://wttr.in/Saanich");

# ╔═╡ 7b52f376-267c-41e5-8834-f28bb95d30d2
# println(readuntil(fname1, "┌"))

# ╔═╡ 6983e532-c2c1-4779-9f12-f6f86a4aa2cf
md"""
Whenever it's not ambiguous, most Julia packages export their key functions so you can use them without a prefix:
"""

# ╔═╡ be159913-7ff3-4932-a9c7-e192122216e9
mean([1,2,3])

# ╔═╡ ba1a585c-4d0b-40a2-9c49-45b58431e998
md"""
     
You can load most kinds of files using the load function as long as you have the right package installed. Here we'll download a fits file and load it:
"""

# ╔═╡ 797876a8-3ace-4b95-b812-57b4eee90ede
fname2 = Downloads.download("http://www.astro.uvic.ca/~wthompson/astroimages/fits/eagle/673nmos.fits");

# ╔═╡ acf8bbbe-9517-4a68-b293-8ce55a118cfd
# load(fname2)

# ╔═╡ 687077eb-43bc-4ceb-8f11-68028409e5a9
md"""

## Reproducilbility
The package versions that you use are stored in the notebook file. If you send the notebook file to someone else, it will automatically use all the same package versions.
"""

# ╔═╡ 53885b9e-697f-4a2d-83a7-8660706e5a53
md"""
# Arrays

In Julia, lists, vectors, matrices, etc are all just Arrays:
"""

# ╔═╡ 82443935-3f20-49bd-b1f6-963ae007e806
r = [2, 3, 1]

# ╔═╡ 76372a6e-0b5f-47c3-b7ea-db69e81aa784
md"You can push elements to the end of the array. 

The ! isn't anything special. People add it to function names as a convention when that function modifies something."

# ╔═╡ 0900fcf4-f3e3-4d2d-8154-a288fa207418
push!(r, 1)

# ╔═╡ 92fb58c1-0d1b-4fae-bc2a-665b9f01ab13
md"""
Array indexing starts at 1!

The first element is index 1, the second is 2, wow!
"""

# ╔═╡ cd7f74dc-f3ec-44fe-bc7a-0ddc2d38cdaa
q = [1,2,3,4]

# ╔═╡ a98de02c-92ca-4a0a-8de2-92554e5d99cc
q[1]

# ╔═╡ d2730f5f-246d-4a87-9048-1bf93aaf7376
md"""
You can also use begin and end:
"""

# ╔═╡ 1e78f9fc-a397-47ab-a9ba-c8b066c0bfd5
q[begin]

# ╔═╡ 26f81574-10cb-48f2-b96e-efa03605ec34
q[end]

# ╔═╡ f014d8a6-8149-4641-b8f5-a7c064dd79c6
q[begin+2]

# ╔═╡ 45403da6-22f4-442c-b1df-afca1ab00dd1
md"Functions without `!` return new copies of data instead of changing it out from under you:"

# ╔═╡ 80195847-01be-40dd-8aa0-0ae71554e335
# sort!(x)

# ╔═╡ 2722b909-072f-4833-a660-f28bb2cda83c
md"You can make matrices really easily:"

# ╔═╡ 54c2c632-4ea1-40fb-9cbc-996186e26a54
A = [
    1 2 3
    4 5 6
    7 8 9
]

# ╔═╡ a6b13709-9e5c-4a15-a53c-51aa7d2f8d6c
md"Just like MATLAB and Fortran, in Julia 1D arrays are column vectors by default. This is opposite from Python!"

# ╔═╡ ade2f288-141e-432b-ba49-62c2cbf2b670
md"""
Operators like `*`, `^2`, `sin()`, or `exp()` apply to their whole argument (unlike NumPy):
"""

# ╔═╡ 9d60fcb9-98eb-4ccb-aa79-b0ab6ca96bdc
A * [1, 2, 3]

# ╔═╡ 42f9bdb0-f786-4d0e-903e-6d18db1bb80e
A*A

# ╔═╡ 1e69ca99-1902-48a3-b635-f6b12b93c04a
A^2

# ╔═╡ f85dcb95-de89-454b-86f5-96ab81759c36
sin(A)

# ╔═╡ f02b3011-ac3b-4c7f-ad1b-512fd44aeb49
exp(A)

# ╔═╡ ce088a02-580e-4152-9aff-de61885d12d4
md"""
## Broadcasting

If you want to apply an operation *element* wise you can prefix anything (**anything**) with a `.`:
"""

# ╔═╡ 890bd344-1a02-42f6-b984-ed0a7ed7768c
A .* [1, 2, 3]

# ╔═╡ 1b9bc1ba-1e38-4211-abb4-eba4da063ffd
A.^2

# ╔═╡ e5eff35a-1eee-4831-ab99-fc96b0955a70
[1 2 3] .* [
             1
             2
             3 ]

# ╔═╡ 746e6c58-87b1-4039-9d6c-45eaeda71441
sin.(A)

# ╔═╡ 731351ed-4c71-4d1d-888e-d9b3efb9bfdb
begin
	strs = [
	    "A",
	    "B",
	    "C",
	]
	strs2 = ["a" "b"]
	strs .* strs2
end

# ╔═╡ c8215e41-f2b4-430b-aa7a-b69211e4da0b
md"Ranges just store the start, step, and stop, but otherwise work just like any other Array:"

# ╔═╡ fc53a866-eb0b-4550-ae7c-205e0b746174
1:5

# ╔═╡ 784e1160-9218-42d7-95b2-a812caac05de
arr = [1:5; 6:8]

# ╔═╡ a334c418-541f-45cd-b649-a3a2a2c77d82
md"""
## Boolean Masks, Slicing
"""

# ╔═╡ 60e9367a-d5e7-430b-8e74-a434fe02b8e1
aa = randn(10)

# ╔═╡ e1169fd7-ec56-4df7-abb6-74daff1b4346
mask = -0.2 .< aa .< 0.2

# ╔═╡ 7f32d06b-93ca-4c5d-92ce-5a5ba7e52820
aa[mask]

# ╔═╡ dc2b85b9-2d39-4eab-9dcd-7e878f9526e9
md"## If statements"

# ╔═╡ 7b49c472-6970-42a5-8e78-ff923e8ae091
if 2 in (1,2,3,4)
    println("This is an if statement")
end

# ╔═╡ 8c061cef-913c-4219-9eca-3a2193910fcd
md"

!!! note
    We did not put a `:` at the end of the if. The whitespace / indentation doesn't matter, we just have to put `end` at the end.
"

# ╔═╡ 91f918d1-8bbb-482e-b8f4-5cb5fefa368d
if 2 in (1,2,3,4)
    println("This way is fine too")
end

# ╔═╡ e495831e-b443-442f-bf87-52d3a1c87e7a
ϕ = 4

# ╔═╡ aede685d-1994-495c-a42c-b0ca1c592837
if 1 < ϕ < 9
    println("Ranges are supported")
end

# ╔═╡ 7bd5ae1c-2b3c-4fc7-8bdd-ac4f4c15ca9f
md"In julia, we use `&&` instead of `and`, `||` instead of `or`, and `!` instead of `not`."

# ╔═╡ d090ef27-f04d-40c2-93b8-854e8993e43d
if any(!isfinite, 1:5) || false
    println("another example")
end

# ╔═╡ b3734a21-2ccb-462e-bb72-e5e6b9e3cf9f
true || false

# ╔═╡ ecad91ca-2002-4576-ac1c-35f1d2a87459
true && false

# ╔═╡ eed5e58f-9e6f-429f-88cb-5725487683d3
!(true || false)

# ╔═╡ 07ab8a63-ed06-4291-8995-20e3ce960016
md"""
## Loops

Loops are not bad! You can use for loops in your Julia programs as much as you want †, or you can use arrays if that makes more sense for a problem. You are free to choose!

†: If they're inside functions
"""

# ╔═╡ 73986c63-d35b-46a9-9b36-ece278638034
for i in 1:10
    println(i^2)
end

# ╔═╡ de5d9e7f-d53b-4a23-9c4f-6db5101c937e
md"""
Often, processing one element at a time is more efficient than putting everything in arrays...


The reason is that it can reduced memory usage. You don't need to allocate large arrays to store temporary data at every stage, you just store the input and output.
"""

# ╔═╡ 0b6a4d7f-8eac-4f99-b46e-e0729d1c8233
function slow(inputs)
	a = inputs .^2
	b = a ./ 2
	c = b .+ a
	d = sqrt.(c)
	return sum(d)
end

# ╔═╡ 0d9a81e8-c39e-40bd-9d72-09c4efd77e9e
function fast(inputs)
	total = 0.0
	for i in eachindex(inputs)
		a = inputs[i] ^2
		b = a / 2
		c = b + a
		d = sqrt(c)
		total += d
	end
	return total
end

# ╔═╡ d511f086-e483-4b1a-bc5c-2e42a00b9a2c
data = randn(1_000_000)

# ╔═╡ b2b25747-f60b-48fd-b314-675a070b847a
md"""
## Multi-Threading

Using multiple threads to speed up your code is easy in Julia! You can prefix most for loops with `@threads` to get an automatic speedup. This does require you to start Julia with more than one thread (-t auto usually).
"""

# ╔═╡ 8d5a0019-df72-47b5-8f81-4a1a1dadbeb9
md"""
## Structs

You can create your own *objects* types using `struct`. These are really efficient and a great way to structure your program.

Many times, you should start with structs instead of arrays.
"""

# ╔═╡ 4ea25781-bb11-413b-9992-cb26b0206767
struct MyPoint
    x::Float64
    y::Float64
    z::Float64
end

# ╔═╡ 537ff2e8-34cf-44ff-87e4-dd8379404b9c
p1 = MyPoint(1,2,3)

# ╔═╡ b3c6bbdd-9a92-41da-92f2-5c6093d5bf89
p1.x

# ╔═╡ 8b1e558d-c004-4650-963d-b83aa42b00d5
md"We can define methods of built in functions like - (minus)"

# ╔═╡ d9d62b02-be32-4a9f-b582-a386f4e720dc
Base.:-(p1::MyPoint, p2::MyPoint) = MyPoint(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

# ╔═╡ a554679b-1ec6-4037-86b9-4c5991097b18
@time slow(data)

# ╔═╡ 1cb9edac-1a09-4914-911b-90f6b01a5693
@time fast(data)

# ╔═╡ db31390e-4beb-4b51-803b-b5580252615f
@time for i in 1:10#00
    sum(rand(1000,1000).^2)
end

# ╔═╡ c4245a63-20ba-4ae8-8f6c-0df222e153db
@time Threads.@threads for i in 1:10#00
    sum(rand(1000,1000).^2)
end

# ╔═╡ 6e7cde6e-94d7-4c21-9c81-9bc0016311ca
md"And now we can do math with our points:"

# ╔═╡ 5989f9ed-0440-4a7d-b76d-a8135ecd3c87
p2 = MyPoint(1,3,4)

# ╔═╡ bd6cb511-1132-413c-a53e-1c0ec0a2f045
p2 - p1

# ╔═╡ d95979af-004d-450b-a341-ebb4b8f52255
[p1, p2] .- [p2, p1]

# ╔═╡ f6177244-b48a-44ce-8ae1-b257d56929cf
md"Or even define new operators:"

# ╔═╡ b58afb78-803b-4d14-aa23-a6c7a7c93d11
⨳(p1::MyPoint, p2::MyPoint) = p1.x / p2.y + p1.y / p2.z + p1.z / p2.x

# ╔═╡ 78e10084-9f38-41bb-9d94-96af97d91177
p1 ⨳ p2

# ╔═╡ d9e5bc1d-c8e4-4482-927f-2e52a30e6d47
md"""
Of course this works for any function! It doesn't have to be an operator/symbol. This mechanism of defining standard functions for your own types (known as adding new methods) is the main reason why Julia packages work so well together!

There's no `np.cos`, `math.cos`, `sympy.cos`, `jax.cos`, ..., there's just `cos`!
"""

# ╔═╡ 56233d74-bac5-4576-9d43-a1be0dea8fe2
md"""
## Automatic Differentiation

Most Julia code can be differentiated just like any other mathematical expression using an autodiff library. Here, we'll use ForwardDiff.jl.

This is a super power: if you have a forward modelling code and you want to compare it to data, you can get not just the  but also the gradient of that  with respect to all your model parameters. This can make your modelling code even more efficient!
"""

# ╔═╡ 66bca807-727e-4fec-92e4-877177192068
gaussmodel(x, μ, A, σ) = A * exp(-(x-μ)^2/σ)

# ╔═╡ 7eb34502-14fb-4232-9e38-07ef63a25a59
xdat = 0:0.5:3

# ╔═╡ c5f030b3-aece-439a-a659-cc8ae13efc8e
dat = sin.(xdat)

# ╔═╡ 5d901ed9-a56b-46d5-904c-7dd3de0aea85
meansquare(d1, d2) = sqrt(mean((d1 .- d2).^2))

# ╔═╡ a8b27e6b-a524-4a8c-aa88-ba4893820a23
fit((μ, A, σ)) = meansquare(gaussmodel.(xdat, μ, A, σ), dat)

# ╔═╡ 0d3c8bf1-047d-4aa8-ba46-abb7eed02158
fit((1, 0.1, 2))

# ╔═╡ 3c8b3902-df2e-4ae7-ae20-805bc651549b
ForwardDiff.gradient(fit, [1, 0.1, 2])

# ╔═╡ 96bb49d1-34f0-4077-9271-73af26056f0a
ForwardDiff.hessian(fit, [1, 0.1, 2])

# ╔═╡ 8d3d85e0-0261-4df7-ada9-fe480cb6653d
md"""
## Uncertainty

The Measurements package let's you propagate uncertainty assuming Gaussian distributed errors and linear error propagataion. You can combine it with lots of other packages like Unitful for physical units with uncertainties, or simulations to propagate uncertainties through your code.

Have calculations where these assumptions don't fit? Try MonteCarloMeasurements.jl
"""

# ╔═╡ ad60cc4d-20f2-440f-b07f-726e48f182cb
am = 2 ± 1

# ╔═╡ be5a873c-150b-43ef-85f7-e538462b2a6c
bm = 4 ± 2

# ╔═╡ bc5c26b8-e7a1-43eb-9437-fe024d6d7251
am * bm

# ╔═╡ 9e5f7e2e-c019-4759-88c2-833183f96ef9
mean([am, bm])

# ╔═╡ 6fc1b7c5-d9f1-4f60-8ee4-3cd91b3d0aa8
md"The package IntervalArithmetic.jl is also worth mentioning. It let's you propagate an interval e.g. 1..2 through your calculations and get an output interval garuanteed to include any values between [1,2]. This is a great way to test your code for floating point round off errors!"

# ╔═╡ 84371e24-4bac-4f75-94a3-c28549e1026d
md"""
## Data

Julia has great libraries for working with tabular data. You can easily load:

* CSV
* Numpy .npz
* Excel
* R data
* MATLAB .mat
* SQL
* Arrow

"""

# ╔═╡ 114f6327-239d-4ac0-b863-10a97d1636b1
df = DataFrame(
    "A" =>  1:10,
    "B" => 11:20
)

# ╔═╡ 494cdeeb-3d76-4ebd-a6d3-7714c7b281be
md"Let's load the Hipparcos-Gaia Catalog of Accelerations by Tim Brandt (2021) as a CSV file:"

# ╔═╡ 25b3bdd5-490f-425b-a35b-62fec36c4732
Downloads.download("https://people.sc.fsu.edu/~jburkardt/data/csv/hurricanes.csv","hurricanes.csv")

# ╔═╡ debd2776-160e-436a-9eeb-9a0feab8e82c
hur = CSV.read("hurricanes.csv", DataFrame)

# ╔═╡ 8313fa25-bb4c-46de-967c-60f94924af4e
hur.Average

# ╔═╡ f024db20-0222-4388-8ec4-86cc96191c4a
sort(hur, "Month")

# ╔═╡ 6d94d0f2-0297-47da-86e8-f6f7ccdeac37
md"""
## Plotting

Julia has a new plotting library, called `Makie`:

> The name Makie (we pronounce it Mah-kee) is derived from the japanese word Maki-e, which is a technique to sprinkle lacquer with gold and silver powder. Data is the gold and silver of our age, so let's spread it out beautifully on the screen!


You can also just use matplotlib via PyPlot.jl.
"""

# ╔═╡ 80b960fa-8aeb-4466-a092-e7e9704cef71
# using CairoMakie; CairoMakie.activate!()

# ╔═╡ 7abc8099-42f2-4e0f-aa72-211903713257
lines(1:180, sind)

# ╔═╡ 58cb2375-2469-41ac-abb0-27d5801ccde4
xx = 1:100

# ╔═╡ c69cc078-9880-4d54-ad8f-e0a6cafe333d
yy = atan.(xx)

# ╔═╡ 5b582e3b-0ba7-475e-9f7e-6fea7f1ed766
scatter(xx, yy)

# ╔═╡ 4010d5e5-4107-4019-8f35-fcb55045afeb
let
	x = range(-π, π, length=100)
	y = range(-π, π, length=100)
	z = sinc.(sqrt.(x.^2 .+ y'.^2))
	
	surface(x, y, z, colormap=:plasma)
end

# ╔═╡ f8b9c7c3-039d-435c-93eb-cdb2af093dcc
let
	fig = Figure(
	    size=(800,800)
	)
	
	xx = π/4*(-1:0.001:1)
	yy = π/4*(-1:0.001:1)
	zz = sinc.(sqrt.(xx.^2 .+ yy'.^2))
	
	ax1 = Makie.Axis(fig[1,1], xlabel="x", ylabel="y")
	scatter!(ax1, xx, zz[:,1], )
	
	ax2 = Makie.Axis(fig[2,1], xlabel="x", ylabel="y")
	h = contourf!(ax2, xx, yy, zz)
	
	Colorbar(fig[1:2, 2], h, label="Colorbar")
	
	
	linkxaxes!(ax1, ax2)
	linkyaxes!(ax1, ax2)
	
	fig
end

# ╔═╡ b72f7451-366c-49ca-bdb6-2f2eab4175e5
md"""
## Plotting our DataFrame

Returning to dataframe, let us plot it.
"""

# ╔═╡ 68df69b5-b143-4e02-80e9-316a2823ccd9
let
	fig = Figure()
	ax = Axis(
		fig[1,1],
		xticks = (1.0:length(hur.Month), hur.Month)
	)
	for year in string.(2005:2015)
		scatterlines!(ax, 
			eachindex(hur.Month),
			hur[:,year],
			label=year
		)
	end
	Legend(fig[1,2],ax)
	fig
end

# ╔═╡ a852b53d-a002-4475-8cd4-003393d8fc9d
hur

# ╔═╡ 15d737c5-b7db-41c0-b339-07151d995022
md"""
## Statistical Modelling

The package Turing.jl offers a powerful language for doing Bayesian modelling. The syntax  is a little uncommon in astronomy, but in Statistics it's a standard way of saying that  is a random variable with the probability distribution of .

Like most Julia packages, Turing can be mixed with other packages to create something better than the sum of it's parts. For example, you can combine Turing with the Flux.jl deep learning library to create Bayesian neural nets.

We'll start by generating some simulated data of a linear relationship:
"""

# ╔═╡ 2b797359-62f6-4607-a673-0937b5431f33
begin
	N = 40
	x = 20rand(N)
	m = 0.4
	B = 4
	y = m .* x .+ B .+ 2randn(N)
	
	scatter(
	    x, y,
	    axis=(
	        xlabel="x",
	        ylabel="y",
	    )
	)
end

# ╔═╡ 2e200325-5358-4deb-bc16-5954905415fd
sortedx = sort(x)

# ╔═╡ e5187373-4325-4eaa-b09f-6d8f9f362772
# ╠═╡ disabled = true
#=╠═╡
using PythonCall
  ╠═╡ =#

# ╔═╡ 6b3b1c56-8edf-47cd-9209-70ba69b7147b
# Define a simple Normal model with unknown mean and variance.
@model function linear_regression_1(x, y)

    # Priors on slope, itercept, and variance of data around the line
    m  ~ Normal(sqrt(10))
    σ₂ ~ truncated(Normal(0, 100), lower=0)
    β  ~ Normal(0, sqrt(3))

    # Equation of a line
    μ = m .* x .+ β

    # We model our y points has being drawn from a Normal distribution about that line
    y ~ MvNormal(μ, sqrt(σ₂))
end

#  Run sampler, collect results

# ╔═╡ 99da649b-c29e-41ba-8d89-478b1fe023c4
lin_model = linear_regression_1(x, y);

# ╔═╡ c90ac1a2-27a7-4c1b-b48d-048f823ab0c3
chain = sample(lin_model, NUTS(0.85), 5_000);

# ╔═╡ e3abfe15-a239-4ed5-aa67-22da6c9aeafe
chain

# ╔═╡ d2875071-dbf8-4647-85a2-ede7907226de
# Make a traceplot
lines(chain["m"][:])

# ╔═╡ 912aa996-36b1-4dc5-b94d-5d3d131e44be


# ╔═╡ 3ac99e68-e090-46bc-bf7b-a4d7813417a5
md"Let's visualize the posterior:"

# ╔═╡ ccc4f412-a823-4922-a512-ef1d3b09795b
let
	# We'll plot our posterior samples as lines with 200 steps between 0 and 20
	xpost = range(-5, 25, length=200)
	
	# Grab 300 posterior samples at random
	ii = rand(eachindex(chain["m"]), 300)
	# ypost = chain["m"][ii] .* xpost' .+ chain["β"][ii]

	fig = Figure()
	ax = Axis(fig[1,1],
		xlabel="x",
		ylabel="y",
	)

	pl = Makie.ablines!(
		chain["β"][ii],
		chain["m"][ii],
		color=:black,
		alpha=0.02,
	    label="posterior",
	)
	
	# Overplot our data
	s = scatter!(ax, x, y, label="data")
	
	Legend(fig[1,2], [s,pl], ["data","posterior"], "Legend")
	
	xlims!(ax, low=-2, high=22)
	
	fig

end

# ╔═╡ 268c5458-0442-4360-a9e8-ef7dc2950bba
md"We can also make a corner plot 😄"

# ╔═╡ 9056d64f-df24-4484-bfb0-5bd45ccf33fd
pairplot(chain)

# ╔═╡ f92c51b3-2909-4030-ac87-7501a9e37870
md"""
A few things we've made with Julia 😄

* pairplots and pairplots.py
* octofitter and octofitterpy
* SPIDERS rtc

"""

# ╔═╡ b23104be-cf58-4723-876a-061031a4b9e9
md"""
Using Python libraries

You can access python libraries using the `PythonCall` package:
"""

# ╔═╡ f3bbbf7c-dcf4-4376-b3d9-791c83e43efa


# ╔═╡ b3ebeb5a-d8c5-44bb-882f-ee0b6363662b
pymath = pyimport("numpy");

# ╔═╡ 46813a1b-3e5f-45c7-92eb-1b10763869ce
pymath.sin(1)

# ╔═╡ ff6b198b-df15-44b4-91c8-4aeea07e261f


# ╔═╡ 56bbdd9b-c483-418a-a311-52946899f3c9


# ╔═╡ b7c3964e-8866-4dd5-b86d-00457ec1ae7c


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AstroImages = "fe3fc30c-9b16-11e9-1c73-17dabf39f4ad"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
AstroImages = "~0.4.2"
BenchmarkTools = "~1.5.0"
CSV = "~0.10.14"
DataFrames = "~1.6.1"
ForwardDiff = "~0.10.36"
Measurements = "~2.11.0"
PairPlots = "~2.7.2"
PythonCall = "~0.9.19"
Turing = "~0.31.1"
WGLMakie = "~0.9.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "7577d794fe40551a98fc49ea6a47849d7ee15397"

[[deps.ADTypes]]
git-tree-sha1 = "016833eb52ba2d6bea9fcb50ca295980e728ee24"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.7"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "b0489adc45a7c8cf0d8e2ddf764f89c1c3decebd"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.2.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "Accessors", "DensityInterface", "Random"]
git-tree-sha1 = "6380a9a03a4207bac53ac310dd3a283bb4df54ef"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.8.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "dfa0e3508fc3df81d28624b328f3b937c1df8bc2"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.6.1"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "16589dbdd36c782ff01700908e962b303474f641"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.1"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Random", "Random123", "Requires", "StatsFuns"]
git-tree-sha1 = "672f7ce648e06f93fceefde463c5855d77b6915a"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.5.4"
weakdeps = ["Libtask"]

    [deps.AdvancedPS.extensions]
    AdvancedPSLibtaskExt = "Libtask"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "1f919a9c59cf3dfc68b64c22c453a2e356fca473"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.4"

[[deps.AliasTables]]
deps = ["Random"]
git-tree-sha1 = "ca95b2220ef440817963baa71525a8f2f4ae7f8f"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.0.0"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AstroAngles]]
git-tree-sha1 = "41621fa5ed5f7614b75eea8e0b3cfd967b284c87"
uuid = "5c4adb95-c1fc-4c53-b4ea-2a94080c53d2"
version = "0.1.3"

[[deps.AstroImages]]
deps = ["AbstractFFTs", "AstroAngles", "ColorSchemes", "DimensionalData", "FITSIO", "FileIO", "ImageAxes", "ImageBase", "ImageIO", "ImageShow", "MappedArrays", "PlotUtils", "PrecompileTools", "Printf", "RecipesBase", "Statistics", "Tables", "UUIDs", "WCS"]
git-tree-sha1 = "09dd0aed7460a51d2f35af92255a4f572b8c2a19"
uuid = "fe3fc30c-9b16-11e9-1c73-17dabf39f4ad"
version = "0.4.2"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "490e739172eb18f762e68dc3b928cad2a077983a"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.1"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRules", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "c12f0e123d21091546d11164d2665e70558a32b3"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.13.11"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "965cf07801c88a92c28a61a65c8e9cc67978126c"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "3.0.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CFITSIO]]
deps = ["CFITSIO_jll"]
git-tree-sha1 = "8425c47db102577eefb93cb37b4480e750116b0d"
uuid = "3b1b4be9-1499-4b22-8d78-7db3344d1961"
version = "1.4.1"

[[deps.CFITSIO_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b90d32054fc88f97dd926022f554180e744e4d7d"
uuid = "b3e40c51-02ae-5482-8a39-3ace5868dcf4"
version = "4.4.0+0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3e79289d94b579d81618f4c7c974bb9390dab493"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.64.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"
weakdeps = ["InverseFunctions"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "TOML"]
git-tree-sha1 = "e81c4263c7ef4eca4d645ef612814d72e9255b41"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.22"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "d4e9dc4c6106b8d44e40cd4faf8261a678552c7c"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.12"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DimensionalData]]
deps = ["Adapt", "ArrayInterface", "ConstructionBase", "Dates", "Extents", "IntervalSets", "IteratorInterfaceExtensions", "LinearAlgebra", "PrecompileTools", "Random", "RecipesBase", "SparseArrays", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "8a6e9c0ac3a861b983af862cefabc12519884a13"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.24.13"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "22c595ca4146c07b16bcf9c8bea86f731f7109d2"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.108"
weakdeps = ["ChainRulesCore", "DensityInterface", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "f4dd7727b07b4b7fff5ff4149118ee06e83dfab7"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.55"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MacroTools", "OrderedCollections", "Random", "Requires", "Test"]
git-tree-sha1 = "cb96969530b2617c06cab67374071c610bf70aa7"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.25.2"

    [deps.DynamicPPL.extensions]
    DynamicPPLChainRulesCoreExt = ["ChainRulesCore"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLReverseDiffExt = ["ReverseDiff"]
    DynamicPPLZygoteRulesExt = ["ZygoteRules"]

    [deps.DynamicPPL.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FITSIO]]
deps = ["CFITSIO", "Printf", "Reexport", "Tables"]
git-tree-sha1 = "a8924c203d66d4c5d72980572c6810213422a59d"
uuid = "525bcba6-941b-5504-bd06-fd0dc1a4d2eb"
version = "0.17.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "881275fc6b8c6f0dfb9cfa4a878979a33cb26be3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.1"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d3e63d9fa13f8eaa2f06f64949e2afc593ff52c2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.10"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "801aef8228f7f04972e596b09d4dba481807c913"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "6f93a83ca11346771a93bbde2bdad2f65b61498f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2c3ec1f90bb4a8f7beafb0cffea8a4c3f4e636ab"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "23ddd329f4a2a65c7a55b91553b60849bd038575"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.11"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "896385798a8d49a255c398bd49162062e4a4c435"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.13"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "eb3edce0ed4fa32f75a0a11217433c31d56bd48b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.0"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "ed7167240f40e62d97c1f5f7735dea6de3cc5c49"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.18"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "839c82932db86740ae729779e610f07a1640be9a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.6.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "88b916503aac4fb7f701bb625cd84ca5dd1677bc"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.29+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LRUCache]]
git-tree-sha1 = "b3cc6698599b10e652832c2f23db3cab99d51b59"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.1"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "ed1f362b3fd13f00b65e61d98669c652c17663ab"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.7"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Primes", "SimplePolynomials"]
git-tree-sha1 = "d76cec8007ec123c2b681269d40f94b053473fcf"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.7"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems", "Requires", "SimpleUnPack"]
git-tree-sha1 = "98cad2db1c46f2fff70a5e305fb42c97a251422a"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.9.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "d28056379864318172ff4b7958710cfddd709339"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.6"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8ba8b1840d3ab5b38e7c71c23c3193bb5cbc02b5"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.10"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "d2a45e1b5998ba3fdfb6cfe0c81096d4c7fb40e7"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.9.6"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "46ca613be7a1358fb93529726ea2fc28050d3ae0"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.9"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "Requires"]
git-tree-sha1 = "bdcde8ec04ca84aef5b124a17684bf3b302de00e"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.11.0"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "5055845dd316575ae2fc1f6dcb3545ff15fe547a"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.14"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "0ae91efac93c3859f5c812a24c9468bb9e50b028"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.1"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6572fe0c5b74431aaeb0b18a4aa5ef03c84678be"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.3"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.PairPlots]]
deps = ["Contour", "Distributions", "KernelDensity", "LinearAlgebra", "Makie", "Measures", "Missings", "NamedTupleTools", "OrderedCollections", "PolygonOps", "PrecompileTools", "Printf", "RecipesBase", "Requires", "StaticArrays", "Statistics", "StatsBase", "TableOperations", "Tables"]
git-tree-sha1 = "16f9ed8f3591e2dfc2969b319fc63167cc4fad32"
uuid = "43a3c2be-4208-490b-832a-a21dcd55d7da"
version = "2.7.2"

    [deps.PairPlots.extensions]
    MCMCChainsExt = "MCMCChains"
    PairPlotsDynamicQuantitiesExt = "DynamicQuantities"
    PairPlotsDynamicUnitfulExt = "Unitful"

    [deps.PairPlots.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "81a2a9462003a423fdc59e2a3ff84cde93c4637b"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.7"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "0fe6664f742903eab8929586af78e10a51b33577"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.19"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "d8f131090f2e44b145084928856a561c83f43b27"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.13.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["Accessors", "ChainRulesCore", "CommonSolve", "Printf"]
git-tree-sha1 = "1ab580704784260ee5f45bffac810b152922747b"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.1.5"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "d15c65e25615272e1b1c5edb1d307484c7942824"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.31.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Setfield", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "10499f619ef6e890f3f4a38914481cc868689cd5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.8"

[[deps.SciMLStructures]]
git-tree-sha1 = "5833c10ce83d690c124beedfe5f621b50b02ba4d"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.1.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e182b9e5afb194142d4668536345a365ea19363a"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.2"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymbolicIndexingInterface]]
deps = ["MacroTools", "RuntimeGeneratedFunctions"]
git-tree-sha1 = "f7b1fc9fc2bc938436b7684c243be7d317919056"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.11"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.Tracker]]
deps = ["Adapt", "ChainRulesCore", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5158100ed55411867674576788e710a815a0af02"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.34"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Accessors", "Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "47e516e2eabd0cf1304cd67839d9a85d52dd659d"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.81"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "4fbc97172c751557b81431e7f7c9419de5eabbcb"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.31.1"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.WCS]]
deps = ["ConstructionBase", "WCS_jll"]
git-tree-sha1 = "858cf2784ff27d908df7a3fe22fcd5fbf02f508b"
uuid = "15f3aee2-9e10-537f-b834-a6fb8bdb944d"
version = "0.6.2"

[[deps.WCS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "947bfa11fcd65dac9e9b2e963504fba6b4971d31"
uuid = "550c8279-ae0e-5d1b-948f-937f2608a23e"
version = "7.7.0+0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "bd7da7625f4da33ed48587b4d400f3c38d550038"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.9.9"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "b4a5a3943078f9fd11ae0b5ab1bdbf7718617945"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.5.8+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─8424af6a-01b2-11ef-3775-6b696ebd1762
# ╟─70e030c4-592a-4df2-aa65-9b09c9808408
# ╠═4aaaed7d-7e6d-43a4-acf0-77bd0e7cb088
# ╠═c9be86ac-e5a8-46d5-8619-8bb39c236416
# ╠═708dea70-8e9e-4adb-bbfc-b19acbf22394
# ╟─7454ba00-5671-42ac-aac5-94b6ea75b4eb
# ╠═fb9dc1d0-f277-4c38-a289-603b5f53cbd3
# ╠═42ea29e3-b48e-4f12-bf40-f24fd2407bd5
# ╟─f820fee7-dea8-4329-8bae-ddbae130e802
# ╟─cf265d34-740d-4e14-8bdd-2db70d9e5966
# ╟─6fa7c392-bfbb-4bb1-958c-693e08a95266
# ╟─18781b6c-7a89-44de-8c8b-7f72d8bc9b47
# ╟─2401c963-0bba-42ae-adb8-39a5cd455ad9
# ╠═4d5351d9-c208-4c02-88f2-2d962e23f03a
# ╠═578b0480-53d3-4c38-bff6-4d68fa341f7c
# ╠═e7ab3818-b6b7-46bd-a10f-5c0f1e079364
# ╟─75505abe-c4ad-48c4-ae09-7f07f133e8a1
# ╠═d9e838ed-f5b2-4673-a2c5-a7be6b9e0793
# ╠═98651864-d363-4ad0-ab11-323bf81b9031
# ╟─4837a067-da0d-4e99-9998-5ece7258fd10
# ╠═77559e19-c77b-4f2b-9108-e1991c1d31a7
# ╠═0a48c9d6-d6d6-4c7c-96de-3f6c4bfe64c6
# ╟─b99279cb-65cb-4a9f-82a7-b5cac6f06849
# ╠═b67ff7c5-674b-4b26-aee8-dd00b37526ca
# ╠═057f1d04-6629-4059-ab41-4def9e5dd565
# ╟─fae56d7e-de38-4d8a-bc75-7cb57d9b1467
# ╠═5aebc02a-85e0-43e5-a136-77c7abbdabba
# ╟─15322d81-9104-47b0-b65e-ef0f4ff078bd
# ╠═6a847a64-d3a1-499f-aae4-7754fb5cae63
# ╟─17871d0b-d248-43b8-aa0e-91e931afdee1
# ╠═1852fa12-3af6-454f-861a-60168fe33581
# ╠═66ee9d25-d1b3-4cd2-9f80-f9821d558ad6
# ╠═a5af4194-e39e-42c0-ac80-a0e592dd348c
# ╟─29d53a10-737a-4311-8d46-6b92ef058a51
# ╟─b1db2c82-f835-4089-8b95-5b9a31ee42ec
# ╠═00b6f7ef-6f55-40f4-a14f-1cfbd41215ac
# ╠═08afdbd6-20cd-402e-b21d-cff682287fd6
# ╠═7b52f376-267c-41e5-8834-f28bb95d30d2
# ╟─6983e532-c2c1-4779-9f12-f6f86a4aa2cf
# ╠═90a32f1e-27e0-489f-bf88-b029926ef0ee
# ╠═be159913-7ff3-4932-a9c7-e192122216e9
# ╟─ba1a585c-4d0b-40a2-9c49-45b58431e998
# ╠═76bf1286-fcd1-4969-9c51-d7b8b4bcc138
# ╠═797876a8-3ace-4b95-b812-57b4eee90ede
# ╠═acf8bbbe-9517-4a68-b293-8ce55a118cfd
# ╟─687077eb-43bc-4ceb-8f11-68028409e5a9
# ╟─53885b9e-697f-4a2d-83a7-8660706e5a53
# ╠═82443935-3f20-49bd-b1f6-963ae007e806
# ╟─76372a6e-0b5f-47c3-b7ea-db69e81aa784
# ╠═0900fcf4-f3e3-4d2d-8154-a288fa207418
# ╟─92fb58c1-0d1b-4fae-bc2a-665b9f01ab13
# ╠═cd7f74dc-f3ec-44fe-bc7a-0ddc2d38cdaa
# ╠═a98de02c-92ca-4a0a-8de2-92554e5d99cc
# ╟─d2730f5f-246d-4a87-9048-1bf93aaf7376
# ╠═1e78f9fc-a397-47ab-a9ba-c8b066c0bfd5
# ╠═26f81574-10cb-48f2-b96e-efa03605ec34
# ╠═f014d8a6-8149-4641-b8f5-a7c064dd79c6
# ╟─45403da6-22f4-442c-b1df-afca1ab00dd1
# ╠═2e200325-5358-4deb-bc16-5954905415fd
# ╠═80195847-01be-40dd-8aa0-0ae71554e335
# ╟─2722b909-072f-4833-a660-f28bb2cda83c
# ╠═54c2c632-4ea1-40fb-9cbc-996186e26a54
# ╟─a6b13709-9e5c-4a15-a53c-51aa7d2f8d6c
# ╟─ade2f288-141e-432b-ba49-62c2cbf2b670
# ╠═9d60fcb9-98eb-4ccb-aa79-b0ab6ca96bdc
# ╠═42f9bdb0-f786-4d0e-903e-6d18db1bb80e
# ╠═1e69ca99-1902-48a3-b635-f6b12b93c04a
# ╠═f85dcb95-de89-454b-86f5-96ab81759c36
# ╠═f02b3011-ac3b-4c7f-ad1b-512fd44aeb49
# ╟─ce088a02-580e-4152-9aff-de61885d12d4
# ╠═890bd344-1a02-42f6-b984-ed0a7ed7768c
# ╠═1b9bc1ba-1e38-4211-abb4-eba4da063ffd
# ╠═e5eff35a-1eee-4831-ab99-fc96b0955a70
# ╠═746e6c58-87b1-4039-9d6c-45eaeda71441
# ╠═731351ed-4c71-4d1d-888e-d9b3efb9bfdb
# ╟─c8215e41-f2b4-430b-aa7a-b69211e4da0b
# ╠═fc53a866-eb0b-4550-ae7c-205e0b746174
# ╠═784e1160-9218-42d7-95b2-a812caac05de
# ╟─a334c418-541f-45cd-b649-a3a2a2c77d82
# ╠═60e9367a-d5e7-430b-8e74-a434fe02b8e1
# ╠═e1169fd7-ec56-4df7-abb6-74daff1b4346
# ╠═7f32d06b-93ca-4c5d-92ce-5a5ba7e52820
# ╟─dc2b85b9-2d39-4eab-9dcd-7e878f9526e9
# ╠═7b49c472-6970-42a5-8e78-ff923e8ae091
# ╟─8c061cef-913c-4219-9eca-3a2193910fcd
# ╠═91f918d1-8bbb-482e-b8f4-5cb5fefa368d
# ╠═e495831e-b443-442f-bf87-52d3a1c87e7a
# ╠═aede685d-1994-495c-a42c-b0ca1c592837
# ╟─7bd5ae1c-2b3c-4fc7-8bdd-ac4f4c15ca9f
# ╠═d090ef27-f04d-40c2-93b8-854e8993e43d
# ╠═b3734a21-2ccb-462e-bb72-e5e6b9e3cf9f
# ╠═ecad91ca-2002-4576-ac1c-35f1d2a87459
# ╠═eed5e58f-9e6f-429f-88cb-5725487683d3
# ╟─07ab8a63-ed06-4291-8995-20e3ce960016
# ╠═73986c63-d35b-46a9-9b36-ece278638034
# ╠═de5d9e7f-d53b-4a23-9c4f-6db5101c937e
# ╠═0b6a4d7f-8eac-4f99-b46e-e0729d1c8233
# ╠═0d9a81e8-c39e-40bd-9d72-09c4efd77e9e
# ╟─d511f086-e483-4b1a-bc5c-2e42a00b9a2c
# ╠═a554679b-1ec6-4037-86b9-4c5991097b18
# ╠═1cb9edac-1a09-4914-911b-90f6b01a5693
# ╟─b2b25747-f60b-48fd-b314-675a070b847a
# ╠═db31390e-4beb-4b51-803b-b5580252615f
# ╠═c4245a63-20ba-4ae8-8f6c-0df222e153db
# ╟─8d5a0019-df72-47b5-8f81-4a1a1dadbeb9
# ╠═4ea25781-bb11-413b-9992-cb26b0206767
# ╠═537ff2e8-34cf-44ff-87e4-dd8379404b9c
# ╠═b3c6bbdd-9a92-41da-92f2-5c6093d5bf89
# ╟─8b1e558d-c004-4650-963d-b83aa42b00d5
# ╠═d9d62b02-be32-4a9f-b582-a386f4e720dc
# ╟─6e7cde6e-94d7-4c21-9c81-9bc0016311ca
# ╠═5989f9ed-0440-4a7d-b76d-a8135ecd3c87
# ╠═bd6cb511-1132-413c-a53e-1c0ec0a2f045
# ╠═d95979af-004d-450b-a341-ebb4b8f52255
# ╟─f6177244-b48a-44ce-8ae1-b257d56929cf
# ╠═b58afb78-803b-4d14-aa23-a6c7a7c93d11
# ╠═78e10084-9f38-41bb-9d94-96af97d91177
# ╟─d9e5bc1d-c8e4-4482-927f-2e52a30e6d47
# ╟─56233d74-bac5-4576-9d43-a1be0dea8fe2
# ╠═71ea979a-2ce8-490b-886a-686b3917d34a
# ╠═66bca807-727e-4fec-92e4-877177192068
# ╠═7eb34502-14fb-4232-9e38-07ef63a25a59
# ╠═c5f030b3-aece-439a-a659-cc8ae13efc8e
# ╠═5d901ed9-a56b-46d5-904c-7dd3de0aea85
# ╠═a8b27e6b-a524-4a8c-aa88-ba4893820a23
# ╠═0d3c8bf1-047d-4aa8-ba46-abb7eed02158
# ╠═3c8b3902-df2e-4ae7-ae20-805bc651549b
# ╠═96bb49d1-34f0-4077-9271-73af26056f0a
# ╟─8d3d85e0-0261-4df7-ada9-fe480cb6653d
# ╠═03378a27-5375-4359-b48c-42359f2d113d
# ╠═ad60cc4d-20f2-440f-b07f-726e48f182cb
# ╠═be5a873c-150b-43ef-85f7-e538462b2a6c
# ╠═bc5c26b8-e7a1-43eb-9437-fe024d6d7251
# ╠═9e5f7e2e-c019-4759-88c2-833183f96ef9
# ╟─6fc1b7c5-d9f1-4f60-8ee4-3cd91b3d0aa8
# ╟─84371e24-4bac-4f75-94a3-c28549e1026d
# ╠═2bc67983-9907-47a1-9695-8970726b957f
# ╠═114f6327-239d-4ac0-b863-10a97d1636b1
# ╠═494cdeeb-3d76-4ebd-a6d3-7714c7b281be
# ╠═3fd8a8b1-adbc-4bc0-bf64-6c668c35a67c
# ╠═25b3bdd5-490f-425b-a35b-62fec36c4732
# ╠═debd2776-160e-436a-9eeb-9a0feab8e82c
# ╠═8313fa25-bb4c-46de-967c-60f94924af4e
# ╠═f024db20-0222-4388-8ec4-86cc96191c4a
# ╟─6d94d0f2-0297-47da-86e8-f6f7ccdeac37
# ╠═10292915-20b0-4838-a77e-258249b025b8
# ╠═80b960fa-8aeb-4466-a092-e7e9704cef71
# ╠═7abc8099-42f2-4e0f-aa72-211903713257
# ╠═58cb2375-2469-41ac-abb0-27d5801ccde4
# ╠═c69cc078-9880-4d54-ad8f-e0a6cafe333d
# ╠═5b582e3b-0ba7-475e-9f7e-6fea7f1ed766
# ╠═4010d5e5-4107-4019-8f35-fcb55045afeb
# ╠═f8b9c7c3-039d-435c-93eb-cdb2af093dcc
# ╠═b72f7451-366c-49ca-bdb6-2f2eab4175e5
# ╠═68df69b5-b143-4e02-80e9-316a2823ccd9
# ╠═a852b53d-a002-4475-8cd4-003393d8fc9d
# ╟─15d737c5-b7db-41c0-b339-07151d995022
# ╠═2b797359-62f6-4607-a673-0937b5431f33
# ╠═548c1c6f-a752-4af9-8724-63ad8b3473ad
# ╠═e5187373-4325-4eaa-b09f-6d8f9f362772
# ╠═6b3b1c56-8edf-47cd-9209-70ba69b7147b
# ╠═99da649b-c29e-41ba-8d89-478b1fe023c4
# ╠═c90ac1a2-27a7-4c1b-b48d-048f823ab0c3
# ╠═e3abfe15-a239-4ed5-aa67-22da6c9aeafe
# ╠═d2875071-dbf8-4647-85a2-ede7907226de
# ╠═912aa996-36b1-4dc5-b94d-5d3d131e44be
# ╟─3ac99e68-e090-46bc-bf7b-a4d7813417a5
# ╠═ccc4f412-a823-4922-a512-ef1d3b09795b
# ╟─268c5458-0442-4360-a9e8-ef7dc2950bba
# ╠═69998f08-1202-4220-a539-089fc947f80f
# ╠═9056d64f-df24-4484-bfb0-5bd45ccf33fd
# ╟─f92c51b3-2909-4030-ac87-7501a9e37870
# ╠═b23104be-cf58-4723-876a-061031a4b9e9
# ╠═f3bbbf7c-dcf4-4376-b3d9-791c83e43efa
# ╠═02206a11-b3fe-4fb7-b0a0-5243527d744f
# ╠═b3ebeb5a-d8c5-44bb-882f-ee0b6363662b
# ╠═46813a1b-3e5f-45c7-92eb-1b10763869ce
# ╠═ff6b198b-df15-44b4-91c8-4aeea07e261f
# ╠═56bbdd9b-c483-418a-a311-52946899f3c9
# ╠═b7c3964e-8866-4dd5-b86d-00457ec1ae7c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
