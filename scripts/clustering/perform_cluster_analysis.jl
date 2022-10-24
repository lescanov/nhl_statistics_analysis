# October 23, 2022

using Pkg, OhMyREPL
using CSV, DataFrames, DataFramesMeta, MultivariateStats, Revise, StatsBase, Clustering, Distances, PlotlyJS, UMAP

# I want to identify which clusters of NHL players based on their in-game performance
# The workflow of this project is to form clusters > publish on web as dashboard or with genie
# To start, I will use the 2018-2019 statistics taken from moneypuck

# First, reading in dataset
nhl_stats = CSV.read("data/2018-2019_skater_statistics.csv", DataFrame)

#### PERFORMING PRELIMINARY UMAP ANALYSIS ####
module perform_umap

export umap_5v5
using DataFrames, DataFramesMeta, UMAP, StatsBase, Distances, PlotlyJS
import Main.nhl_stats

# This module will focus on producing numerous UMAPS using different n_neighbor inputs
# This is taken from https://umap-learn.readthedocs.io/en/latest/clustering.html
# In particular, to make UMAP suitable for clustering, they change n_neighbors to equal 30 and min_dist to 0.
# I want to visualize these differences

# Selecting only numeric statistics, convering to matrix of n_features, n_samples
stats_5v5 = @chain nhl_stats begin
    @subset :situation .== "5on5"
    select(Not(Between(:playerId, :situation)))
    permutedims
    Matrix
end

# Standardizing dataframe
stats_5v5 = standardize(ZScoreTransform, stats_5v5, dims = 2)

# Performing umap with different n_neighbors
n_neighbor_paramters = range(5, step = 5, stop = 30)
umap_objects_by_differing_neighbors = []
for neighbors in n_neighbor_paramters
    push!(
        umap_objects_by_differing_neighbors,
        umap(stats_5v5; min_dist = 0.1, n_neighbors = neighbors)
    )
end

# Converting to a dataframe
umap_dataframes = []
for index in umap_objects_by_differing_neighbors
    df = DataFrame(index, :auto)
    df = permutedims(df)
    df = rename(df, :x1 => :UMAP1, :x2 => :UMAP2)
    push!(umap_dataframes, df)
end

# Plotting by neighbors
# defining plotting function that uses PlotlyJS
function plot_umap(df)
    plot(
        df,
        :UMAP1,
        :UMAP2,
        type = "scatter",
        mode = "markers"
    )
end

# Plotting UMAPs with different n_neighbors
# n = 5
plot_umap(umap_dataframes[1])
# n = 10
plot_umap(umap_dataframes[2])
# n = 15
plot_umap(umap_dataframes[3])
# n = 20
plot_umap(umap_dataframes[4])
# n = 25
plot_umap(umap_dataframes[5])
# n = 30
plot_umap(umap_dataframes[6])

# The reduction at n = 30 looks most interested, will continue with it
umap_5v5 = umap_dataframes[6]

# End of module
end

#### PERFORMING K-MEANS CLSUTERING ####
module perform_kmeans_clustering

using DataFrames, DataFramesMeta, Clustering, PlotlyJS
import Main.nhl_stats

# 