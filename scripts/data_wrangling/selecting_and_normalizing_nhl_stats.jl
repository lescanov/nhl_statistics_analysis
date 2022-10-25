"""
Selecting raw data for analysis.
"""

#### Selecting raw statistics from 2018-2019 season ####
module selecting_nhl_stats
export raw_stats

# Loading libraries
using CSV, DataFrames, DataFramesMeta

# Reading in NHL statistics for 2018-2019
nhl_stats = CSV.read("data/2018-2019_skater_statistics.csv", DataFrame)

# Selecting only raw statistics
raw_stats = @chain nhl_stats begin
    select(
        :situation,
        Between(:games_played, :shifts),
        Between(:I_F_primaryAssists, :I_F_dZoneGiveaways),
        Between(:I_F_shifts, :shotsBlockedByPlayer),
        Between(:OnIce_F_shotsOnGoal, :OnIce_F_highDangerGoals),
        :OnIce_F_unblockedShotAttempts,
        Between(:OnIce_A_shotsOnGoal, :OnIce_A_highDangerShots),
        Between(:OnIce_A_lowDangerGoals, :OnIce_A_highDangerGoals),
        :OnIce_A_unblockedShotAttempts
    )
    @subset :situation .== "all"
    select(Not(:situation))
end
end

####  Performing PCA on nhl stats ####
module performing_pca
export nhl_transformed

# Loading libraries
using Main.selecting_nhl_stats
using MultivariateStats, Plots, StatsBase, DataFrames, DataFramesMeta

# Converting dataframe into matrix of n_samples, n_features
nhl_matrix = @chain raw_stats begin
    permutedims
    Matrix
end

# Following analysis is borrowed heavily from:
# https://stackoverflow.com/questions/68053860/exploratory-pca-in-julia

# Standardizing matrix using zscore
# dims = 2 is a row-wise standardization
nhl_matrix = standardize(ZScoreTransform, nhl_matrix, dims = 2)

# Running PCA for nhl stats
nhl_pca = fit(PCA, nhl_matrix, maxoutdim = 3)

# Extracting principal components
nhl_transformed = projection(nhl_pca)' * (nhl_matrix .- mean(nhl_pca))

end

#### PERFORMING K-MEANS CLUSTERING ####
module perform_k_means_clustering

# Loading libraries
using Clustering, Distances, Statistics
using Main.performing_pca

# Performing k-means Clustering
# Using silhouettes to pick a specific k for Clustering
# Calculations taken from documentation of k-means as well as this discussion
# https://discourse.julialang.org/t/clustering-jl-silhouettes-distances/33810
silhouette_values = []
for i in 2:20
    kmeans_cluster = kmeans(nhl_transformed, i; maxiter = 200)
    a = assignments(kmeans_cluster)
    c = counts(kmeans_cluster)
    m = kmeans_cluster.centers
    distances = pairwise(SqEuclidean(), nhl_transformed)
    push!(silhouette_values, mean(silhouettes(a, c, distances)))
end
# The largest silhouette value correspond to a cluster of 2
nhl_clusters = kmeans(nhl_transformed, 2; maxiter = 200, display = :iter)

# Plotting silhouette values

end