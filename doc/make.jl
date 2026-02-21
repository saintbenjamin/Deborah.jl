# ============================================================================
# doc/make.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

ENV["DOCUMENTER_DEBUG"] = "true"

using Documenter, Deborah

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

linkcheck_ignore = [
    # We'll ignore links that point to GitHub's edit pages, as they redirect to the
    # login screen and cause a warning:
    r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)",
    "https://nvd.nist.gov/vuln/detail/CVE-2018-16487",
    # We'll ignore the links to Documenter tags in CHANGELOG.md, since when you tag
    # a release, the release link does not exist yet, and this will cause the linkcheck
    # CI job to fail on the PR that tags a new release.
    r"https://github.com/JuliaDocs/Documenter.jl/releases/tag/v1.\d+.\d+",
]
# Extra ones we ignore only on CI.
if get(ENV, "GITHUB_ACTIONS", nothing) == "true"
    # It seems that CTAN blocks GitHub Actions?
    push!(linkcheck_ignore, "https://ctan.org/pkg/minted")
end

makedocs(
    modules = [Deborah],
    format = if "pdf" in ARGS
        Documenter.LaTeX(
            platform = "native"
        )
    else
        Documenter.HTML(
            sidebar_sitename = false,
            edit_link = nothing
        )
    end,
    build = ("pdf" in ARGS) ? "build-pdf" : "build",
    debug = ("pdf" in ARGS),
    authors = "Benjamin J. Choi",
    sitename = "Deborah.jl",
    linkcheck = "linkcheck" in ARGS,
    linkcheck_ignore = linkcheck_ignore,
    pages = [
        "Home" => "index.md",
        # "Manual" => Any[
        #     "DeborahCore"         => "man/DeborahCore.md",
        # ],
        "Reference" => Any[
            "DeborahCore"         => Any[
                "DeborahCore"                           => "lib/DeborahCore.md",
                "DeborahCore.BaselineSequence"          => "lib/DeborahCore/BaselineSequence.md",
                "DeborahCore.DatasetPartitionerDeborah" => "lib/DeborahCore/DatasetPartitionerDeborah.md",
                "DeborahCore.DeborahRunner"             => "lib/DeborahCore/DeborahRunner.md",
                "DeborahCore.FeaturePipeline"           => "lib/DeborahCore/FeaturePipeline.md",
                "DeborahCore.MLInputPreparer"           => "lib/DeborahCore/MLInputPreparer.md",
                "DeborahCore.MLSequence"                => "lib/DeborahCore/MLSequence.md",
                "DeborahCore.MLSequenceLasso"           => "lib/DeborahCore/MLSequenceLasso.md",
                "DeborahCore.MLSequenceLightGBM"        => "lib/DeborahCore/MLSequenceLightGBM.md",
                "DeborahCore.MLSequenceMiddleGBM"       => "lib/DeborahCore/MLSequenceMiddleGBM.md",
                "DeborahCore.MLSequencePyCallLightGBM"  => "lib/DeborahCore/MLSequencePyCallLightGBM.md",
                "DeborahCore.MLSequenceRidge"           => "lib/DeborahCore/MLSequenceRidge.md",
                "DeborahCore.PathConfigBuilderDeborah"  => "lib/DeborahCore/PathConfigBuilderDeborah.md",
                "DeborahCore.ResultPrinterDeborah"      => "lib/DeborahCore/ResultPrinterDeborah.md",
                "DeborahCore.SummaryWriterDeborah"      => "lib/DeborahCore/SummaryWriterDeborah.md",
                "DeborahCore.TOMLConfigDeborah"         => "lib/DeborahCore/TOMLConfigDeborah.md",
                "DeborahCore.XYMLInfoGenerator"         => "lib/DeborahCore/XYMLInfoGenerator.md",
                "DeborahCore.XYMLVectorizer"            => "lib/DeborahCore/XYMLVectorizer.md",
            ],
            "DeborahDocument"     => Any[
                "DeborahDocument"       => "lib/DeborahDocument.md",
                "DeborahDocumentRunner" => "lib/DeborahDocument/DeborahDocumentRunner.md",
            ],
            "DeborahEsther"       => Any[
                "DeborahEsther"                         => "lib/DeborahEsther.md",
                "DeborahEsther.DeborahEstherRunner"     => "lib/DeborahEsther/DeborahEstherRunner.md",
                "DeborahEsther.EstherDependencyManager" => "lib/DeborahEsther/EstherDependencyManager.md",
            ],
            "DeborahEstherMiriam" => Any[
                "DeborahEstherMiriam"                           => "lib/DeborahEstherMiriam.md",
                "DeborahEstherMiriam.DeborahEstherMiriamRunner" => "lib/DeborahEstherMiriam/DeborahEstherMiriamRunner.md",
                "DeborahEstherMiriam.MiriamDependencyManager"   => "lib/DeborahEstherMiriam/MiriamDependencyManager.md",
                "DeborahEstherMiriam.MiriamExistenceManager"    => "lib/DeborahEstherMiriam/MiriamExistenceManager.md",
            ],
            "DeborahThreads"      => Any[
                "DeborahThreads"       => "lib/DeborahThreads.md",
                "DeborahThreadsRunner" => "lib/DeborahThreads/DeborahThreadsRunner.md",
            ],
            "Elijah"              => Any[
                "Elijah"                     => "lib/Elijah.md",
                "DeborahWizardRunner"        => "lib/Elijah/DeborahWizardRunner.md",
                "DeborahThreadsWizardRunner" => "lib/Elijah/DeborahThreadsWizardRunner.md",
                "EstherWizardRunner"         => "lib/Elijah/EstherWizardRunner.md",
                "EstherThreadsWizardRunner"  => "lib/Elijah/EstherThreadsWizardRunner.md",
                "MiriamWizardRunner"         => "lib/Elijah/MiriamWizardRunner.md",
                "MiriamThreadsWizardRunner"  => "lib/Elijah/MiriamThreadsWizardRunner.md",
            ],
            "Esther"              => Any[
                "Esther"                            => "lib/Esther.md",
                "Esther.BootstrapDerivedCalculator" => "lib/Esther/BootstrapDerivedCalculator.md",
                "Esther.DatasetPartitionerEsther"   => "lib/Esther/DatasetPartitionerEsther.md",
                "Esther.EstherRunner"               => "lib/Esther/EstherRunner.md",
                "Esther.JackknifeRunner"            => "lib/Esther/JackknifeRunner.md",
                "Esther.PathConfigBuilderEsther"    => "lib/Esther/PathConfigBuilderEsther.md",
                "Esther.QMomentCalculator"          => "lib/Esther/QMomentCalculator.md",
                "Esther.ResultPrinterEsther"        => "lib/Esther/ResultPrinterEsther.md",
                "Esther.SingleCumulant"             => "lib/Esther/SingleCumulant.md",
                "Esther.SingleQMoment"              => "lib/Esther/SingleQMoment.md",
                "Esther.SummaryWriterEsther"        => "lib/Esther/SummaryWriterEsther.md",
                "Esther.TOMLConfigEsther"           => "lib/Esther/TOMLConfigEsther.md",
                "Esther.TraceDataLoader"            => "lib/Esther/TraceDataLoader.md",
                "Esther.TraceRescaler"              => "lib/Esther/TraceRescaler.md",
            ],
            "EstherDocument"      => Any[
                "EstherDocument"       => "lib/EstherDocument.md",
                "EstherDocumentRunner" => "lib/EstherDocument/EstherDocumentRunner.md",
            ],
            "EstherThreads"       => Any[
                "EstherThreads"       => "lib/EstherThreads.md",
                "EstherThreadsRunner" => "lib/EstherThreads/EstherThreadsRunner.md",
            ],
            "Miriam"              => Any[
                "Miriam"                         => "lib/Miriam.md",
                "Miriam.Cumulants"               => "lib/Miriam/Cumulants.md",
                "Miriam.CumulantsBundle"         => "lib/Miriam/CumulantsBundle.md",
                "Miriam.CumulantsBundleUtils"    => "lib/Miriam/CumulantsBundleUtils.md",
                "Miriam.Ensemble"                => "lib/Miriam/Ensemble.md",
                "Miriam.EnsembleUtils"           => "lib/Miriam/EnsembleUtils.md",
                "Miriam.FileIO"                  => "lib/Miriam/FileIO.md",
                "Miriam.Interpolation"           => "lib/Miriam/Interpolation.md",
                "Miriam.MiriamRunner"            => "lib/Miriam/MiriamRunner.md",
                "Miriam.MultiEnsembleLoader"     => "lib/Miriam/MultiEnsembleLoader.md",
                "Miriam.PathConfigBuilderMiriam" => "lib/Miriam/PathConfigBuilderMiriam.md",
                "Miriam.Reweighting"             => "lib/Miriam/Reweighting.md",
                "Miriam.ReweightingBundle"       => "lib/Miriam/ReweightingBundle.md",
                "Miriam.ReweightingCurve"        => "lib/Miriam/ReweightingCurve.md",
                "Miriam.ReweightingCurveBundle"  => "lib/Miriam/ReweightingCurveBundle.md",
                "Miriam.TOMLConfigMiriam"        => "lib/Miriam/TOMLConfigMiriam.md",
                "Miriam.WriteBSOutput"           => "lib/Miriam/WriteBSOutput.md",
                "Miriam.WriteJKOutput"           => "lib/Miriam/WriteJKOutput.md",
            ],
            "MiriamDocument"      => Any[
                "MiriamDocument"       => "lib/MiriamDocument.md",
                "MiriamDocumentRunner" => "lib/MiriamDocument/MiriamDocumentRunner.md",
            ],
            "MiriamThreads"       => Any[
                "MiriamThreads"       => "lib/MiriamThreads.md",
                "MiriamThreadsRunner" => "lib/MiriamThreads/MiriamThreadsRunner.md",
            ],
            "Rahab"               => Any[
                "Rahab"                      => "lib/Rahab.md",
                "Rahab.BlockBinScan"         => "lib/Rahab/BlockBinScan.md",
                "Rahab.CorrPlot"             => "lib/Rahab/CorrPlot.md",
                "Rahab.HistogramOrigML"      => "lib/Rahab/HistogramOrigML.md",
                "Rahab.ObservableHistory"    => "lib/Rahab/ObservableHistory.md",
                "Rahab.ZeroTemperatureScale" => "lib/Rahab/ZeroTemperatureScale.md",
            ],
            "Rebekah"             => Any[
                "Rebekah"               => "lib/Rebekah.md",
                "Rebekah.Comparison"    => "lib/Rebekah/Comparison.md",
                "Rebekah.Heatmaps"      => "lib/Rebekah/Heatmaps.md",
                "Rebekah.JLD2Loader"    => "lib/Rebekah/JLD2Loader.md",
                "Rebekah.JLD2Saver"     => "lib/Rebekah/JLD2Saver.md",
                "Rebekah.PXvsBSPlotter" => "lib/Rebekah/PXvsBSPlotter.md",
                "Rebekah.PyPlotLaTeX"   => "lib/Rebekah/PyPlotLaTeX.md",
                "Rebekah.SummaryLoader" => "lib/Rebekah/SummaryLoader.md",
            ],
            "RebekahMiriam"       => Any[
                "RebekahMiriam"                              => "lib/RebekahMiriam.md",
                "RebekahMiriam.ComparisonRebekahMiriam"      => "lib/RebekahMiriam/ComparisonRebekahMiriam.md",
                "RebekahMiriam.HeatmapsRebekahMiriam"        => "lib/RebekahMiriam/HeatmapsRebekahMiriam.md",
                "RebekahMiriam.JLD2SaverRebekahMiriam"       => "lib/RebekahMiriam/JLD2SaverRebekahMiriam.md",
                "RebekahMiriam.PXvsBSPlotterRebekahMiriam"   => "lib/RebekahMiriam/PXvsBSPlotterRebekahMiriam.md",
                "RebekahMiriam.ReweightingPlotRebekahMiriam" => "lib/RebekahMiriam/ReweightingPlotRebekahMiriam.md",
                "RebekahMiriam.SummaryLoaderRebekahMiriam"   => "lib/RebekahMiriam/SummaryLoaderRebekahMiriam.md",
            ],
            "Sarah"               => Any[
                "Sarah"                    => "lib/Sarah.md",
                "Sarah.AvgErrFormatter"    => "lib/Sarah/AvgErrFormatter.md",
                "Sarah.BlockSizeSuggester" => "lib/Sarah/BlockSizeSuggester.md",
                "Sarah.Bootstrap"          => "lib/Sarah/Bootstrap.md",
                "Sarah.BootstrapDataInit"  => "lib/Sarah/BootstrapDataInit.md",
                "Sarah.BootstrapRunner"    => "lib/Sarah/BootstrapRunner.md",
                "Sarah.ControllerCommon"   => "lib/Sarah/ControllerCommon.md",
                "Sarah.DataLoader"         => "lib/Sarah/DataLoader.md",
                "Sarah.DatasetPartitioner" => "lib/Sarah/DatasetPartitioner.md",
                "Sarah.Jackknife"          => "lib/Sarah/Jackknife.md",
                "Sarah.JobLoggerTools"     => "lib/Sarah/JobLoggerTools.md",
                "Sarah.NameParser"         => "lib/Sarah/NameParser.md",
                "Sarah.SeedManager"        => "lib/Sarah/SeedManager.md",
                "Sarah.StringTranscoder"   => "lib/Sarah/StringTranscoder.md",
                "Sarah.SummaryCollector"   => "lib/Sarah/SummaryCollector.md",
                "Sarah.SummaryFormatter"   => "lib/Sarah/SummaryFormatter.md",
                "Sarah.TOMLLogger"         => "lib/Sarah/TOMLLogger.md",
                "Sarah.XYInfoGenerator"    => "lib/Sarah/XYInfoGenerator.md",
            ]
        ]
    ],
    checkdocs = :none,
    clean = false,
    warnonly = ("strict=false" in ARGS),
    doctest = ("doctest=only" in ARGS) ? :only : true,
)

# ============================================================================
# Deploy docs
# ============================================================================

if "pdf" in ARGS
    # Move only the generated PDF into a dedicated commit directory
    pdf_commit_dir = joinpath(@__DIR__, "build-pdf", "commit")
    mkpath(pdf_commit_dir)

    for f in readdir(joinpath(@__DIR__, "build-pdf"))
        if endswith(f, ".pdf")
            mv(
                joinpath(@__DIR__, "build-pdf", f),
                joinpath(pdf_commit_dir, f),
                force = true,
            )
        end
    end

    deploydocs(
        repo   = "github.com/saintbenjamin/Deborah.jl.git",
        target = "build-pdf/commit",
        branch = "gh-pages-pdf",
        devbranch = "main",
        forcepush = true,
    )

else
    deploydocs(
        repo   = "github.com/saintbenjamin/Deborah.jl.git",
        branch = "gh-pages",
        devbranch = "main",
        target    = "build",
        forcepush = true,
    )
end