from . import (
    coreg,
    coverage,
    examples,
    hypsometry,
    overview,
    surge,
    surge_examples,
    tcorr_examples,
    uncertainty,
    zonal,
)


def make_all_figures(show: bool = False):
    print("Generating rigidcoreg histogram figure")
    coreg.plot_rigidcoreg_histograms(show=show)

    print("Generating vertcoreg summary figure")
    coreg.plot_vertcoreg_summary(show=show)

    print("Generating annual DEM count figure")
    coverage.plot_annual_dem_count(show=show)

    print("Generating per-pixel source map")
    coverage.plot_source_map(show=show)

    print("Generating terrain error figure")
    uncertainty.plot_terrain_err(show=show)

    print("Generating surge bar figure")
    surge.plot_surge_nosurge_bar(show=show)

    print("Generating surge comparison figures")
    surge.plot_surging_vs_nonsurging_figures(show=show)

    print("Generating post-surge anomaly figure")
    surge.plot_postsurge_anomaly_by_time(show=show)

    print("Generating surge examples figure")
    surge_examples.plot_surge_examples(show=show)

    print("Generating temporal correction examples figure")
    tcorr_examples.plot_tcorr_examples(show=show)

    print("Generating zonal dH/dt figure")
    zonal.plot_zone_dhdt_fig(show=show)

    print("Generating patch method vs uncertainty figure")
    uncertainty.plot_patch_method_vs_vgm(show=show)

    print("Generating baseline error variogram figure")
    uncertainty.plot_baseline_err_variogram(show=show)

    print("Generating hypsometry figure")
    hypsometry.plot_hypsometric_profiles(show=show)

    print("Generating elevation change examples figure")
    examples.plot_elevation_change_examples(show=show)

    # This figure takes time to generate, so it's nice to have that last.
    print("Generating overview dH/dt figure")
    overview.dhdt_overview_fig(show=show)
