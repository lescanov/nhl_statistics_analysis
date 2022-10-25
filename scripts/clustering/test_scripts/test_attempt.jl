#=
Performing PCA analysis and identifying clusters on NHL stats. 
Working with stats in specific contexts, such as 5v5, Powerplay, Penalty kill.
=#

using Pkg, OhMyREPL, CSV, DataFrames, DataFramesMeta, Plots, MultivariateStats, Revise, StatsBase, Clustering, Distances, StatsPlots, PlotlyJS

# First importing raw statistics from 2018-2019
nhl_stats = CSV.read("data/2018-2019_skater_statistics.csv", DataFrame)

# Selecting non-adjusted statistics
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
end

module cluster_stats_5on5

#=
Purpose of this code chunk is to perform PCA and k-means clustering on players stratified by 5v5.
I am selecting only raw statistics 
=#

# Selecting 5v5 statistics and converting to a matrix of n_features, n_samples
stats_5v5 = @chain raw_stats begin
    @subset(:situation .== "5on5", :games_played .>= 10)
    select(Not(:situation))
end

# Attempting to normalize based on time games_played
# Removing games_played so that this does not get normalized
normalized_5v5 = @chain stats_5v5 begin
    select(Not([:games_played, :icetime]))
end

# Finding out how many 60 minute chunks a player had played
player_icetime = stats_5v5.icetime

# Normalizing statistics to per 60 output
for column in range(1, length = ncol(normalized_5v5))
    normalized_5v5[!, column] = (normalized_5v5[!, column] ./ player_icetime) * 60
end

# Converting normalized_5v5 to matrix of n_features, n_samples
matrix_5v5 = @chain normalized_5v5 begin
    permutedims
    Matrix
end

# Using z-scores to normalize matrix
standardized_5v5 = standardize(ZScoreTransform, matrix_5v5, dims = 2)

# Running PCA
pca_5v5 = fit(PCA, standardized_5v5, maxoutdim = 3)

# First do components explain only about 69% of variance
# Now will transform PCA components
transformed_5v5 = projection(pca_5v5)' * (matrix_5v5 .- mean(pca_5v5))

# Performing muliple K-means cluster and utilizing silhouettes to determine optimal k
silhouette_values = []
for i in 2:20
    kmeans_cluster = kmeans(transformed_5v5, i; maxiter = 200)
    a = assignments(kmeans_cluster)
    c = counts(kmeans_cluster)
    m = kmeans_cluster.centers
    distances = pairwise(SqEuclidean(), transformed_5v5)
    push!(silhouette_values, mean(silhouettes(a, c, distances)))
end

# Looking at largest silhouette value
largest_silhouette_value = 0
value_of_k = 2
for (index, value) in enumerate(silhouette_values)
    if value > largest_silhouette_value
        largest_silhouette_value = silhouette_values[index]
        value_of_k = index + 1 # We start out with 2 clusters by default
    end
end
@show largest_silhouette_value value_of_k

# To be honest, this is not what I had hoped for, since there is only 2 clsuters
# But moving forward now with this analysis.
# Performing k means clustering with k of 2
kmeans_5v5 = kmeans(transformed_5v5, 2;  maxiter = 200, display = :iter)

# Bringing it all together into a single dataframe
# Retrieving k means assignments
kmeans_5v5_assignments = assignments(kmeans_5v5)

# Retriving principal components from PCA
principal_components_5v5 = MultivariateStats.transform(pca_5v5, matrix_5v5)

# Putting into single dataframe
clustered_5v5 =  @chain principal_components_5v5 begin
    DataFrame(:auto)
    permutedims
    @transform :cluster = kmeans_5v5_assignments
    rename(
        :x1 => :PCA1,
        :x2 => :PCA2,
        :x3 => :PCA3
    )
end

# Now to plot this mess using plots
# With plotly
PlotlyJS.plot(
    clustered_5v5,
    x = :PCA1,
    y = :PCA2,
    z = :PCA3,
    color = :cluster,
    type = "scatter3d",
    mode = "markers"
)

# Plotting with Plots
@with clustered_5v5 begin
    Plots.scatter(
        :PCA1,
        :PCA2,
        :PCA3,
        color = :cluster
    )
end

# End of module
end

#### CLUSTERING WITH ALL DATA ####
module cluster_with_all_data_5v5

# Importing all 5v5 stats and converting to matrix N-features, N-samples
stats_5v5 = @chain nhl_stats begin
    @subset :situation .== "5on5"
    select(Not(Between(:playerId, :situation)))
    permutedims
    Matrix
end

# Standardizing using z-scores
standardized_5v5_matrix = standardize(ZScoreTransform, stats_5v5, dims = 2)

# Performing PCA
standardized_5v5_pca = fit(PCA, standardized_5v5_matrix, maxoutdim = 3)

# Extracting principal components
pc_standardized_5v5 = MultivariateStats.transform(standardized_5v5_pca, standardized_5v5_matrix)

# Now optimizing k-means
silhouette_values = []
for i in 2:700
    kmeans_cluster = kmeans(transformed_5v5, i; maxiter = 200)
    a = assignments(kmeans_cluster)
    c = counts(kmeans_cluster)
    m = kmeans_cluster.centers
    distances = pairwise(SqEuclidean(), transformed_5v5)
    push!(silhouette_values, mean(silhouettes(a, c, distances)))
end

# Finding largets silhouette value
largest_silhouette_value = 0
value_of_k = 2
for (index, value) in enumerate(silhouette_values)
    if value > largest_silhouette_value
        largest_silhouette_value = silhouette_values[index]
        value_of_k = index + 1 # We start out with 2 clusters by default
    end
end
print(largest_silhouette_value, " ", value_of_k)

# Plotting PCA
# First putt into one dataframe
clustered_5v5 =  @chain principal_components_5v5 begin
    DataFrame(:auto)
    permutedims
    @transform :cluster = kmeans_5v5_assignments
    rename(
        :x1 => :PCA1,
        :x2 => :PCA2,
        :x3 => :PCA3
    )
end

# End of module
end