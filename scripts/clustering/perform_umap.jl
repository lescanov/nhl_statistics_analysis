# October 25, 2022

# This function uses UMAP as dimensionality reduction tool.
# Input is expected to be a standardized matrix of n_features, n_samples
# This function has a parameter called num_of_neighbors that specifies
# how many neighboring points should be used in constructing clusters.
# This will project results into 3 dimnesions.
# Consult main UMAP documentation for further details.
# Otherwise, use default paramters by UMAP_ function.

# The output of this function is either a matrix, when return_matrix = true,
# or a dataframe when return_matrix = false.
# A matrix can be used for input with kmeans clustering.
# The dataframe returned has samples as rows, features as columns.
# Since the UMAP is projected to 3 dimensions, columns are called
# UMAP1, 2 and 3 respectively.

##### PERFORMING UMAP ON STANDARDIZED DATAFRAME ####
module PerformUMAP
export perform_umap
using DataFramesMeta, StatsBase, Distances, UMAP

function perform_umap(input_df::DataFrame, num_of_neighbors::Int, return_matrix::Bool)
    # Performing UMAP based on specified parameters
    umap_result = umap(input_df, 3, n_neighbors = num_of_neighbors)

    # Can either return a dataframe or matrix
    # Matrix will be for input into perform_optimal_kmeans
    # DataFrame will be for plotting
    if return_matrix == false
        df_umap = @chain umap_result begin
            DataFrame(:auto)
            permutedims
            rename(:x1 => :UMAP1, :x2 => :UMAP2, :x3 => :UMAP3)
        end
        return(df_umap)
    else 
        return(umap_result)
    end
end

# End of module
end