# October 23, 2022

using Pkg, OhMyREPL, CSV, DataFrames

# I want to identify which clusters of NHL players based on their in-game performance
# The workflow of this project is to form clusters > publish on web as dashboard or with genie
# To start, I will use the 2018-2019 statistics taken from moneypuck

# First, reading in dataset
nhl_stats = CSV.read("data/2018-2019_skater_statistics.csv", DataFrame)

#### PERFORMING PRELIMINARY UMAP ANALYSIS ####
module perform_umap
export umap_5v5
export umap_n30

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
        umap(stats_5v5, 3; min_dist = 0.1, n_neighbors = neighbors)
    )
end

# Converting to a dataframe
umap_dataframes = []
for index in umap_objects_by_differing_neighbors
    df = DataFrame(index, :auto)
    df = permutedims(df)
    df = rename(df, :x1 => :UMAP1, :x2 => :UMAP2, :x3 => :UMAP3)
    push!(umap_dataframes, df)
end

# Plotting by neighbors
# defining plotting function that uses PlotlyJS
function plot_umap(df)
    PlotlyJS.plot(
        df,
        x = :UMAP1,
        y = :UMAP2,
        z = :UMAP3,
        type = "scatter3d",
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

# Exporting also the matrix correspond to n =30
umap_n30 = umap_objects_by_differing_neighbors[6]

# End of module
end

#### PERFORMING K-MEANS CLSUTERING ON RAW DATA ####
module perform_kmeans_clustering_on_raw_data

using DataFrames, DataFramesMeta, Clustering, PlotlyJS, StatsBase, Distances
import Main.nhl_stats
import Main.perform_umap

# Converting dataframe to matrix of n_features, n_samples
stats_5v5 = @chain nhl_stats begin
    @subset :situation .== "5on5"
    select(Not(Between(:playerId, :situation)))
    permutedims
    Matrix
end

# Standardizing dataframe
stats_5v5 = standardize(ZScoreTransform, stats_5v5, dims = 2)

# Determining optimal k value using silhouette values
silhouette_values = []
for i in 2:20
    kmeans_cluster = kmeans(stats_5v5, i; maxiter = 200)
    a = assignments(kmeans_cluster)
    c = counts(kmeans_cluster)
    m = kmeans_cluster.centers
    distances = pairwise(SqEuclidean(), stats_5v5)
    push!(silhouette_values, mean(silhouettes(a, c, distances)))
end

# Identifying largest silhouette value
# When using a global variable inside a for loop with local scope
# you will have to have to delcare it with the global statement
# https://discourse.julialang.org/t/x-not-defined-error-even-though-it-is/20714
largest_silhouette_value = 0
value_of_k = 2
for (index, value) in enumerate(silhouette_values)
    global largest_silhouette_value
    if value > largest_silhouette_value
        largest_silhouette_value = silhouette_values[index]
        value_of_k = index + 1 # We start out with 2 clusters by default
    end
end
@show largest_silhouette_value value_of_k

# Plotting based on optimal k, which is 2
kmeans_5v5 = kmeans(stats_5v5, 2; maxiter = 200)
kmeans_5v5 = assignments(kmeans_5v5)

# Creating a dataframe for plotting
clustered_5v5 = @chain perform_umap.umap_5v5 begin
    @transform :kmeans_cluster = kmeans_5v5
end

# Plotting
plot(clustered_5v5, :UMAP1, :UMAP2, color = :kmeans_cluster, type = "scatter", mode = "markers")

# End of module
end

#### PERFORMING K-MEANS CLUSTERING ON UMAP ####
module perform_kmeans_on_umap

using DataFrames, DataFramesMeta, Clustering, PlotlyJS, Distances, StatsBase
import Main.perform_umap.umap_n30 as umap_n30
import Main.perform_umap.umap_5v5 as umap_df
import Main.nhl_stats as stats

silhouette_values = []
for i in 2:20
    kmeans_cluster = kmeans(umap_n30, i; maxiter = 200)
    a = assignments(kmeans_cluster)
    c = counts(kmeans_cluster)
    m = kmeans_cluster.centers
    distances = pairwise(SqEuclidean(), umap_n30)
    push!(silhouette_values, mean(silhouettes(a, c, distances)))
end

# Identifying largest silhouette value
largest_silhouette_value = 0
value_of_k = 2
for (index, value) in enumerate(silhouette_values)
    global largest_silhouette_value
    if value > largest_silhouette_value
        largest_silhouette_value = silhouette_values[index]
        value_of_k = index + 1 # We start out with 2 clusters by default
    end
end
@show largest_silhouette_value value_of_k

# 3 clusters is the value obtained here, perform k-means with k =3
umap_kmeans = kmeans(umap_n30, 3; maxiter = 200)
umap_kmeans_assignments = assignments(umap_kmeans)
# Adding this to a dataframe with UMAP values
umap_clustered = @chain umap_df begin
    @transform(
        :cluster = umap_kmeans_assignments
    )
end

# Plotting with PlotlyJS
plot(
    umap_clustered,
    x = :UMAP1,
    y = :UMAP2,
    z = :UMAP3,
    color = :cluster,
    type = "scatter3d",
    mode = "markers",
    marker = attr(size = 5)
)

# End of module
end

