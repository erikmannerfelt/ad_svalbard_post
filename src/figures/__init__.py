from . import coreg, hypsometry, overview, surge, timeseries, uncertainty, zonal


def make_all_figures(show: bool = False):
    print("Generating rigidcoreg histogram figure")
    coreg.plot_rigidcoreg_histograms(show=show)

    print("Generating vertcoreg summary figure")
    coreg.plot_vertcoreg_summary(show=show)

    print("Generating terrain error figure")
    uncertainty.plot_terrain_err(show=show)

    print("Generating surge bar figure")
    surge.plot_surge_nosurge_bar(show=show)

    print("Generating surge comparison figures")
    surge.plot_surging_vs_nonsurging_figures(show=show)

    print("Generating zonal dH/dt figure")
    zonal.plot_zone_dhdt_fig(show=show)

    print("Generating patch method vs uncertainty figure")
    uncertainty.plot_patch_method_vs_vgm(show=show)

    print("Generating baseline error variogram figure")
    uncertainty.plot_baseline_err_variogram(show=show)

    print("Generating hypsometry figure")
    hypsometry.plot_hypsometric_profiles(show=show)

    print("Generating overview dH/dt figure")
    overview.dhdt_overview_fig(show=show)

    print("Generating point timeseries trends figure")
    timeseries.plot_point_timeseries_trends(show=show)
