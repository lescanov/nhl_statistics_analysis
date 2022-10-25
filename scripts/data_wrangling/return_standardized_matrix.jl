#### CREATING A STANDARDIZED MATRIX FROM NHL STAT DATAFRAME ####
module StandardizeMatrixFromDataFrame
export return_standardized_matrix
using DataFramesMeta, StatsBase
# This removes non-numeric columns from nhl statistics

function return_standardized_matrix(input_df::DataFrame, situation::String)
    # Defining acceptable situation input
    acceptable_situations = ["All", "5on4", "4on5", "5on5"]

    if situation in acceptable_situations
        # Cleaning up dataframe to yield numeric matrix
        df = @chain input_df begin
            @subset :situation .== situation
            select(Not(Between(:playerId, :situation)))
            permutedims
            Matrix
        end

        # Standardizing matrix
        standardized_df = standardize(ZScoreTransform, df, dims = 2)
        return(standardized_df)
    else
        println("Situation must be of [All, 5on4, 4on5, 5on5]")
    end
end

# End of module
end