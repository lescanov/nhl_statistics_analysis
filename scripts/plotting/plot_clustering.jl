# October 25, 2022

# Purpose of function is to plot a 3d scatter plot clustered by a column.
# Input is a dataframe. The intended input is a clustered dataframe.

#### PLOTTING UMAP FUNCTION ####
module PlottingClusters
export plot_clustering
using DataFramesMeta, PlotlyJS

function plot_clustering(input_df::DataFrame, x_axis::String, y_axis::String, z_axis::String, cluster_colum::String)
    plot(
        input_df,
        x = Symbol(x_axis),
        y = Symbol(y_axis),
        z = Symbol(z_axis),
        color = Symbol(cluster_colum),
        type = "scatter3d",
        mode = "markers",
        marker = attr(size = 5)
    )
end

# End of module
end