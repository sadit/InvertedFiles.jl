using Documenter, InvertedFiles

makedocs(;
    modules=[InvertedFiles],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/InvertedFiles.jl/blob/{commit}{path}#L{line}",
    sitename="InvertedFiles.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/InvertedFiles.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Inv. Files" => "invfile.md",
        "Sparse" => "sparse.md",
    ],
    warnonly = true
)

deploydocs(;
    repo="github.com/sadit/InvertedFiles.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
