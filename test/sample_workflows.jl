sample_root() = normpath(joinpath(@__DIR__, "..", "sample"))

function prepare_sample_config(
    sample_cfg_name::String,
    tempdir::String;
    analysis_header::String="smoke",
)
    sample_cfg_path = joinpath(sample_root(), sample_cfg_name)
    cfg = TOML.parsefile(sample_cfg_path)

    mkpath(tempdir)
    data_root = joinpath(tempdir, "nf4_clover_wilson_finiteT")
    cp(joinpath(sample_root(), "nf4_clover_wilson_finiteT"), data_root; force=true)

    cfg["data"]["location"] = data_root
    cfg["data"]["analysis_header"] = analysis_header

    if haskey(cfg, "bootstrap")
        cfg["bootstrap"]["N_bs"] = 2
        cfg["bootstrap"]["blk_size"] = 1
    end
    if haskey(cfg, "trajectory")
        cfg["trajectory"]["nkappaT"] = min(Int(cfg["trajectory"]["nkappaT"]), 3)
    end

    temp_cfg_path = joinpath(tempdir, basename(sample_cfg_name))
    open(temp_cfg_path, "w") do io
        TOML.print(io, cfg)
    end

    return temp_cfg_path, cfg
end

function assert_sample_files_exist(location::String, ensemble::String, files::Vector{String})
    for fname in unique(files)
        @test isfile(joinpath(location, ensemble, fname))
    end
end

@testset "Sample workflow smoke tests" begin
    mktempdir() do tempdir
        deborah_cfg_path, deborah_cfg_raw = prepare_sample_config(
            "config_Deborah_L12T4b1.60k13575_Plaq-Rect-TrM1_GBM_LBP_30_TRP_30.toml",
            joinpath(tempdir, "deborah"),
        )
        esther_cfg_path, esther_cfg_raw = prepare_sample_config(
            "config_Esther_L12T4b1.60k13575_Plaq-Rect-TrM1_GBM_Plaq-Rect-TrM2_GBM_Plaq-Rect-TrM3_GBM_Plaq-Rect-TrM4_GBM_LBP_30_TRP_30.toml",
            joinpath(tempdir, "esther"),
        )
        miriam_cfg_path, miriam_cfg_raw = prepare_sample_config(
            "config_Miriam_L12T4b1.60_Plaq-Rect-TrM1_GBM_Plaq-Rect-TrM2_GBM_Plaq-Rect-TrM3_GBM_Plaq-Rect-TrM4_GBM_LBP_30_TRP_30.toml",
            joinpath(tempdir, "miriam"),
        )

        @testset "Deborah sample config" begin
            deborah_cfg = TOMLConfigDeborah.parse_full_config_Deborah(deborah_cfg_path)
            x_list = deborah_cfg.data.model == "Baseline" ? [deborah_cfg.data.Y] : deborah_cfg.data.X
            paths = PathConfigBuilderDeborah.build_path_config_Deborah(
                deborah_cfg.data,
                deborah_cfg.abbrev,
                x_list,
            )

            assert_sample_files_exist(
                deborah_cfg.data.location,
                deborah_cfg.data.ensemble,
                vcat(deborah_cfg.data.X, [deborah_cfg.data.Y]),
            )
            @test paths.path == joinpath(deborah_cfg.data.location, deborah_cfg.data.ensemble, "")
            @test startswith(paths.analysis_dir, deborah_cfg.data.location)
            @test occursin("smoke_$(deborah_cfg.data.ensemble)", paths.analysis_dir)
            @test endswith(paths.info_file, ".toml")
            @test isdir(paths.analysis_dir)
        end

        @testset "Deborah -> Esther bridge config" begin
            esther_cfg = TOMLConfigEsther.parse_full_config_Esther(esther_cfg_path)
            esther_paths = PathConfigBuilderEsther.build_path_config_Esther(
                esther_cfg.data,
                esther_cfg.abbrev,
            )

            esther_files = String[]
            append!(esther_files, esther_cfg.data.TrM1_X)
            append!(esther_files, esther_cfg.data.TrM2_X)
            append!(esther_files, esther_cfg.data.TrM3_X)
            append!(esther_files, esther_cfg.data.TrM4_X)
            append!(esther_files, [
                esther_cfg.data.TrM1_Y,
                esther_cfg.data.TrM2_Y,
                esther_cfg.data.TrM3_Y,
                esther_cfg.data.TrM4_Y,
            ])

            assert_sample_files_exist(
                esther_cfg.data.location,
                esther_cfg.data.ensemble,
                esther_files,
            )
            @test esther_paths.path == joinpath(esther_cfg.data.location, esther_cfg.data.ensemble, "")
            @test startswith(esther_paths.analysis_dir, esther_cfg.data.location)
            @test occursin("smoke_$(esther_cfg.data.ensemble)", esther_paths.analysis_dir)
            @test isdir(esther_paths.my_tex_dir)

            abbrev = StringTranscoder.parse_string_dict(esther_cfg_raw["abbreviation"])
            bridge_cfg = EstherDependencyManager.generate_toml_dict(
                esther_cfg.data.location,
                esther_cfg.data.ensemble,
                esther_cfg.data.analysis_header,
                esther_cfg.data.TrM1_X,
                esther_cfg.data.TrM1_Y,
                esther_cfg.data.TrM1_model,
                esther_cfg_raw["data"]["TrM1_read_column_X"],
                esther_cfg_raw["data"]["TrM1_read_column_Y"],
                esther_cfg_raw["data"]["TrM1_index_column"],
                esther_cfg.data.LBP,
                esther_cfg.data.TRP,
                esther_cfg_raw["deborah"]["IDX_shift"],
                esther_cfg_raw["deborah"]["dump_X"],
                esther_cfg.bs.ranseed,
                esther_cfg.bs.N_bs,
                esther_cfg.bs.blk_size,
                esther_cfg.bs.method,
                esther_cfg.jk.bin_size,
                abbrev,
                esther_cfg.data.use_abbreviation,
            )

            @test bridge_cfg["data"]["location"] == esther_cfg.data.location
            @test bridge_cfg["data"]["ensemble"] == esther_cfg.data.ensemble
            @test bridge_cfg["data"]["analysis_header"] == esther_cfg.data.analysis_header
            @test bridge_cfg["data"]["X"] == esther_cfg.data.TrM1_X
            @test bridge_cfg["data"]["Y"] == esther_cfg.data.TrM1_Y
            @test bridge_cfg["bootstrap"]["N_bs"] == esther_cfg.bs.N_bs
            @test bridge_cfg["jackknife"]["bin_size"] == esther_cfg.jk.bin_size
        end

        @testset "Deborah -> Esther -> Miriam bridge config" begin
            miriam_cfg = TOMLConfigMiriam.parse_full_config_Miriam(miriam_cfg_path)
            miriam_paths = PathConfigBuilderMiriam.build_path_config_Miriam(
                miriam_cfg.data,
                miriam_cfg.abbrev,
            )

            miriam_files = String[]
            append!(miriam_files, miriam_cfg.data.TrM1_X)
            append!(miriam_files, miriam_cfg.data.TrM2_X)
            append!(miriam_files, miriam_cfg.data.TrM3_X)
            append!(miriam_files, miriam_cfg.data.TrM4_X)
            append!(miriam_files, [
                miriam_cfg.data.TrM1_Y,
                miriam_cfg.data.TrM2_Y,
                miriam_cfg.data.TrM3_Y,
                miriam_cfg.data.TrM4_Y,
            ])

            for ensemble in miriam_cfg.data.ensembles
                assert_sample_files_exist(
                    miriam_cfg.data.location,
                    ensemble,
                    miriam_files,
                )
            end

            @test startswith(miriam_paths.analysis_dir, miriam_cfg.data.location)
            @test occursin("smoke_$(miriam_cfg.data.multi_ensemble)", miriam_paths.analysis_dir)
            @test isdir(miriam_paths.my_tex_dir)
            @test endswith(miriam_paths.fname.rwt_all_bs, ".dat")

            abbrev = StringTranscoder.parse_string_dict(miriam_cfg_raw["abbreviation"])
            bridge_cfg = MiriamDependencyManager.generate_toml_dict(
                miriam_cfg.data.location,
                miriam_cfg.data.ensembles[1],
                miriam_cfg.data.TrM1_X,
                miriam_cfg.data.TrM1_Y,
                miriam_cfg.data.TrM1_model,
                miriam_cfg_raw["data"]["TrM1_read_column_X"],
                miriam_cfg_raw["data"]["TrM1_read_column_Y"],
                miriam_cfg_raw["data"]["TrM1_index_column"],
                miriam_cfg.data.TrM2_X,
                miriam_cfg.data.TrM2_Y,
                miriam_cfg.data.TrM2_model,
                miriam_cfg_raw["data"]["TrM2_read_column_X"],
                miriam_cfg_raw["data"]["TrM2_read_column_Y"],
                miriam_cfg_raw["data"]["TrM2_index_column"],
                miriam_cfg.data.TrM3_X,
                miriam_cfg.data.TrM3_Y,
                miriam_cfg.data.TrM3_model,
                miriam_cfg_raw["data"]["TrM3_read_column_X"],
                miriam_cfg_raw["data"]["TrM3_read_column_Y"],
                miriam_cfg_raw["data"]["TrM3_index_column"],
                miriam_cfg.data.TrM4_X,
                miriam_cfg.data.TrM4_Y,
                miriam_cfg.data.TrM4_model,
                miriam_cfg_raw["data"]["TrM4_read_column_X"],
                miriam_cfg_raw["data"]["TrM4_read_column_Y"],
                miriam_cfg_raw["data"]["TrM4_index_column"],
                miriam_cfg.data.LBP,
                miriam_cfg.data.TRP,
                miriam_cfg.input_meta.ns,
                miriam_cfg.input_meta.nt,
                miriam_cfg.input_meta.nf,
                miriam_cfg.input_meta.beta,
                miriam_cfg.input_meta.kappa_list[1],
                miriam_cfg.bs.ranseed,
                miriam_cfg.bs.N_bs,
                miriam_cfg.bs.blk_size,
                miriam_cfg.bs.method,
                miriam_cfg.jk.bin_size,
                miriam_cfg.data.analysis_header,
                miriam_cfg_raw["deborah"]["IDX_shift"],
                miriam_cfg_raw["deborah"]["dump_X"],
                abbrev,
                miriam_cfg.data.use_abbreviation,
            )

            @test bridge_cfg["data"]["location"] == miriam_cfg.data.location
            @test bridge_cfg["data"]["ensemble"] == miriam_cfg.data.ensembles[1]
            @test bridge_cfg["data"]["TrM4_Y"] == miriam_cfg.data.TrM4_Y
            @test bridge_cfg["input_meta"]["kappa"] == miriam_cfg.input_meta.kappa_list[1]
            @test bridge_cfg["bootstrap"]["N_bs"] == miriam_cfg.bs.N_bs
            @test bridge_cfg["jackknife"]["bin_size"] == miriam_cfg.jk.bin_size
        end
    end
end
