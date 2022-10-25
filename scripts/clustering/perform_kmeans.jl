# October 25, 2022

# Purpose of this function is to take a standardized matrix, of
# n_features x n_samples, as input, and perform kmeans clustering
# with an optimal k value found via silhouettes.
# It will test k values from 2 to num_k_to_test, which is a int that
# defines the k cutoff to test.

# This function yields a vector of n length, which specifies clusters.

#### PERFORMING KMEANS CLUSTERING ON 
module PerformKmeans
export perform_optimal_kmeans
using DataFramesMeta, Clustering, StatsBase, Distances

function perform_optimal_kmeans(input_matrix::Matrix, num_of_k_to_test::Int)
    # Creating silhouette values for 2:num_of_k_to_test
    silhouette_values = []
    for i in 2:num_of_k_to_test
        kmeans_cluster = kmeans(input_matrix, i; maxiter = 1000)
        a = assignments(kmeans_cluster)
        c = counts(kmeans_cluster)
        distances = pairwise(SqEuclidean(), input_matrix)
        push!(silhouette_values, mean(silhouettes(a, c, distances)))
    end

    # Determining k value that produces largest mean(silhouettes)
    optimal_k = 0
    largest_silhouette = 0
    for (index, value) in enumerate(silhouette_values)
        if value > largest_silhouette
            push!(largest_silhouette, value)
            push!(optimal_k, index + 1)
        end
    end
    @show largest_silhouette, optimal_k

    # Performing clustering with optimal K
    optimal_kmeans = kmeans(input_matrix, optimal_k; maxiter = 1000)

    # Returning a vector of clusters based on optimal K
    optimal_kmeans_clusters = assignments(optimal_kmeans_clusters)
    return(optimal_kmeans_clusters)
end