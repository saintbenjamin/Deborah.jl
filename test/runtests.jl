using Test
using Random
using TOML
using Deborah

const NameParser = Deborah.Sarah.NameParser
const StringTranscoder = Deborah.Sarah.StringTranscoder
const DatasetPartitioner = Deborah.Sarah.DatasetPartitioner
const SeedManager = Deborah.Sarah.SeedManager
const BlockSizeSuggester = Deborah.Sarah.BlockSizeSuggester
const TOMLConfigDeborah = Deborah.DeborahCore.TOMLConfigDeborah
const PathConfigBuilderDeborah = Deborah.DeborahCore.PathConfigBuilderDeborah
const TOMLConfigEsther = Deborah.Esther.TOMLConfigEsther
const PathConfigBuilderEsther = Deborah.Esther.PathConfigBuilderEsther
const TOMLConfigMiriam = Deborah.Miriam.TOMLConfigMiriam
const PathConfigBuilderMiriam = Deborah.Miriam.PathConfigBuilderMiriam
const EstherDependencyManager = Deborah.DeborahEsther.EstherDependencyManager
const MiriamDependencyManager = Deborah.DeborahEstherMiriam.MiriamDependencyManager

@testset "Deborah.jl" begin
    @testset "Top-level smoke test" begin
        @test isdefined(Deborah, :Sarah)
        @test isdefined(Deborah, :run_Deborah)
        @test isdefined(Deborah, :run_Deborah_Esther)
        @test isdefined(Deborah, :run_Deborah_Esther_Miriam)
        @test isdefined(Deborah, :run_DeborahWizard)
        @test isdefined(Deborah, :run_DeborahThreads)
    end

    @testset "NameParser" begin
        @test NameParser.model_suffix("LightGBM") == "GBM"
        @test NameParser.model_suffix("Lasso") == "LAS"
        @test NameParser.model_suffix("Ridge") == "RID"
        @test NameParser.model_suffix("Baseline") == "BAS"
        @test NameParser.model_suffix("Random") == "RGM"
        @test NameParser.model_suffix("MiddleGBM") == "MDG"
        @test NameParser.model_suffix("PyGBM") == "PYG"
        @test_throws ErrorException NameParser.model_suffix("UnknownModel")

        @test NameParser.make_X_Y(["pbp.dat"], "pbp.dat") == "pbp.dat"
        @test NameParser.make_X_Y(["plaq.dat", "rect.dat"], "pbp.dat") == "plaq.dat_rect.dat_pbp.dat"

        @test NameParser.build_trace_name(
            "L12T4b1.60k13575",
            "Plaq-Rect-TrM1",
            ["plaq.dat", "rect.dat"],
            "pbp.dat",
            "30",
            "10",
            "GBM",
        ) == "L12T4b1.60k13575_Plaq-Rect-TrM1_GBM_LBP_30_TRP_10"

        @test NameParser.build_trace_name(
            "L12T4b1.60k13575",
            "TrM1",
            ["pbp.dat"],
            "pbp.dat",
            "30",
            "10",
            "GBM",
        ) == "L12T4b1.60k13575_TrM1_BAS_LBP_30_TRP_10"
    end

    @testset "StringTranscoder" begin
        @test StringTranscoder.parse_string_dict(
            Dict("labels" => 10, "use_abbreviation" => true, "tag" => "GBM"),
        ) == Dict("labels" => "10", "use_abbreviation" => "true", "tag" => "GBM")

        abbrev_cfg = StringTranscoder.abbreviation_map(
            Dict{String, Any}(
                "plaq.dat" => "Plaq",
                "rect.dat" => "Rect",
                "pbp.dat" => "TrM1",
            ),
        )

        encoded = StringTranscoder.input_encoder_abbrev(
            ["plaq.dat", "rect.dat"],
            "pbp.dat",
            abbrev_cfg,
        )
        @test encoded == "Plaq-Rect-TrM1"
        @test StringTranscoder.input_decoder_abbrev(encoded, abbrev_cfg) == "plaq.dat_rect.dat_pbp.dat"

        abbrev_map = Dict(
            "plaq.dat" => "Plaq",
            "rect.dat" => "Rect",
            "pbp.dat" => "TrM1",
        )
        reverse_map = Dict(v => k for (k, v) in abbrev_map)
        @test StringTranscoder.input_encoder_abbrev_dict(
            ["plaq.dat", "rect.dat"],
            "pbp.dat",
            abbrev_map,
        ) == encoded
        @test StringTranscoder.input_decoder_abbrev_dict(encoded, reverse_map) == "plaq.dat_rect.dat_pbp.dat"

        for name in keys(abbrev_cfg.name_to_code)
            code = abbrev_cfg.name_to_code[name]
            idx = abbrev_cfg.name_to_num[name]
            @test abbrev_cfg.code_to_name[code] == name
            @test abbrev_cfg.num_to_name[idx] == name
            @test abbrev_cfg.num_to_code[idx] == code
            @test abbrev_cfg.code_to_num[code] == idx
        end

        @test_throws ErrorException StringTranscoder.input_encoder_abbrev(
            ["missing.dat"],
            "pbp.dat",
            abbrev_cfg,
        )
        @test_throws ErrorException StringTranscoder.input_decoder_abbrev(
            "Plaq-Missing-TrM1",
            abbrev_cfg,
        )
    end

    @testset "DatasetPartitioner" begin
        lb_idx, tr_idx, bc_idx, ul_idx = DatasetPartitioner.gen_set_idx(8, 4, 2, 2, 4, 0)
        @test lb_idx == [1, 3, 5, 7]
        @test tr_idx == [1, 3]
        @test bc_idx == [2, 4]
        @test ul_idx == [2, 4, 6, 8]

        shifted_lb_idx, shifted_tr_idx, shifted_bc_idx, shifted_ul_idx =
            DatasetPartitioner.gen_set_idx(8, 4, 2, 2, 4, 1)
        @test shifted_lb_idx == [2, 4, 6, 8]
        @test shifted_tr_idx == [1, 3]
        @test shifted_bc_idx == [2, 4]
        @test shifted_ul_idx == [1, 3, 5, 7]

        @test DatasetPartitioner.find_equivalent_shift([1, 3, 5, 7], 8, 4, 1) == 0
        @test DatasetPartitioner.find_equivalent_shift([2, 4, 6, 8], 8, 4, 1) == 1
    end

    @testset "BlockSizeSuggester" begin
        sizes = BlockSizeSuggester.suggest_opt_block_sizes(100, 30, 20, 70, 10)
        @test sizes == Dict(:all => 10, :lb => 3, :bc => 2, :ul => 7)

        partition = DatasetPartitioner.DatasetPartitionInfo(
            100,
            100,
            1,
            30,
            10,
            30,
            10,
            20,
            20,
            70,
            70,
            collect(1:30),
            collect(1:10),
            collect(11:30),
            collect(31:100),
        )
        @test BlockSizeSuggester.suggest_opt_block_sizes(partition, 10) == sizes
        @test_throws ErrorException BlockSizeSuggester.suggest_opt_block_sizes(0, 0, 0, 0, 10)
        @test_throws ErrorException BlockSizeSuggester.suggest_opt_block_sizes(100, 30, 20, 70, 0)
    end

    @testset "SeedManager" begin
        pool1 = SeedManager.setup_rng_pool(2026)
        pool2 = SeedManager.setup_rng_pool(2026)

        @test rand(pool1.rng, UInt64) == rand(pool2.rng, UInt64)
        @test rand(pool1.rng_lb, UInt64) == rand(pool2.rng_lb, UInt64)
        @test rand(pool1.rng_tr, UInt64) == rand(pool2.rng_tr, UInt64)
        @test rand(pool1.rng_bc, UInt64) == rand(pool2.rng_bc, UInt64)
        @test rand(pool1.rng_ul, UInt64) == rand(pool2.rng_ul, UInt64)

        rng1 = SeedManager.setup_rng(850528)
        rng2 = SeedManager.setup_rng(850528)
        @test rand(rng1, UInt64) == rand(rng2, UInt64)
    end

    include("sample_workflows.jl")
end
