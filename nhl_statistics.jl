"""
This is my attempt to learn machine learning in Julia!
I want to test the efficacy of NHL advanced statistics in predicting player point totals.
NHL statistics were downloaded from moneypuck.com. 
"""

# Loading packages
using Pkg, CSV, DataFrames, DataFramesMeta, Statistics, Gadfly, UMAP, StatsBase

# Importing 2018-2019 money puck NHL statistics
nhl_stats_1819 = CSV.read("2018-2019_skater_statistics.csv", DataFrame)

"""
It would be an interesting mini-project to determine how the Vancouver Canucks have changed the last 5 years
Specifically, I will create some summary statistics for every team and compare across different positions
I will create several summary statistics for 5on5 play such as:
    1. Mean point totals
    2. Mean point per game totals
    3. Mean on ice corsi %
    4. Mean off ice corsi %
    5. Relative Corsi %
    6. Mean shot attempts
    7. Mean shots on goal
        Further stratify this on shot quality

"""
# Creating a function that produces 
function create_summary_stats(input_df)
    @chain input_df begin
        @transform(
            :point_per_game = :I_F_points ./ :games_played,
            :relative_corsi_percentage = :onIce_corsiPercentage .- :offIce_corsiPercentage
        )
        @subset(:situation .== "5on5", :games_played .>= 30)
        @by(
            [:team, :position, :season],
            :average_point_total = mean(:I_F_points),
            :average_point_per_game = mean(:point_per_game),
            :average_corsi_on_ice_percentage = mean(:onIce_corsiPercentage) .* 100,
            :average_corsi_off_ice_percentage = mean(:offIce_corsiPercentage) .* 100,
            :average_relative_corsi_percentage = mean(:relative_corsi_percentage) .* 100,
            :average_shots_on_goal = mean(:I_F_shotsOnGoal),
            :average_shot_attempts = mean(:I_F_shotAttempts),
            :average_low_danger_shots = mean(:I_F_lowDangerShots),
            :average_medium_danger_shots = mean(:I_F_mediumDangerShots),
            :average_high_danger_shots = mean(:I_F_highDangerShots)
        )
        @orderby :team
    end
end

# Testing function
season_1819_5on5 = create_summary_stats(nhl_stats_1819)

# Now will do a basic exercise in plotting
# I will gadfly to plot this since this supposedly easy

plot(season_1819_5on5, x = :team, y = :average_point_total, color = :team, Geom.boxplot)

# Plotting median average point totals onto gadfly
# Testing formatting of code block as well as @with macro
plot(
    season_1819_5on5,
    x = :team,
    y = :average_point_total,
    color = :team,
    yintercept = [@with(season_1819_5on5, median(:average_point_total))], # Note [median(season_1819_5on5.average_point_total)] works too
    Geom.boxplot,
    Geom.hline(color = "red", style = :dot)
)

# Plotting Average point per game for fowards
plot(
    @subset(season_1819_5on5, :position .!= "D"),
    x = :team,
    y = :average_point_per_game,
    color = :team,
    yintercept = [median(season_1819_5on5.average_point_per_game)],
    Geom.hline(color = "red", style = :dot),
    Geom.boxplot,
    Theme(key_position = :none)
)

# Perhaps interesting to see a correlation with average point total
# and average relative corsi onIce_corsiPercentage
plot(
    season_1819_5on5,
    x = :average_point_per_game,
    y = :average_relative_corsi_percentage,
    color = :team,
    Geom.point,
    Theme(key_position = :none)
)

# The summary statistics are okay for team trends, however are not that great
# to determining the real scope and spread of the players per team.
# Let's look at forwards, 5v5, min 30 games, all 2019_skater_statistics
forwards_5v5 = @chain nhl_stats_1819 begin
    @transform(
        :points_per_game = :I_F_points ./ :games_played,
        :relative_corsi_percentage = (:onIce_corsiPercentage .- :offIce_corsiPercentage) *100
    )
    @subset(:position .!= "D", :games_played .>= 30, :situation .== "5on5")
    @orderby :team
end

# A reminder, this is just points 5v5
plot(
    forwards_5v5,
    y = :points_per_game,
    x = :team,
    color = :team,
    yintercept = [median(forwards_5v5.points_per_game)],
    Geom.hline(color = "red", style = :dot),
    Geom.boxplot

)

# Visualizing relationship bweteen point total points and relative corsi
plot(
    forwards_5v5,
    x = :relative_corsi_percentage,
    y = :points_per_game,
    color = :position,
    Geom.point
)

# Perhaps I can turn this project into evaluating how Corsi predicts point totals
# And perhaps advanced statistics as a whole
# Perhaps this is best used to describe certain players (forwards) at specific
# game contexts (5v5, pk, etc...)

# Perhaps can work on a classification problem, how to identify which players are which
# Could try identifying players based on clustering 
# Clustering I will try is UMAP


# First normalizing input using z-scores
# Removing non-numeric input
umap_input = @chain nhl_stats_1819 begin
    @subset(:situation .== "5on5")
    select(Not([:playerId, :season, :team, :position, :name, :situation]))
end

# Z-score normalization
mapcols!(zscore, umap_input)

# Input for UMAP is matrix of n_features, n_samples
# Transposing dataframe into matrix
umap_input = @chain umap_input begin
    permutedims
    Matrix
end

# This won't work because the input is supposed abstract matrix
model = umap(umap_input)
model_df = DataFrame(model, :auto)

# Defining player names (potentiall useful for umap)
player_names = @chain nhl_stats_1819 begin
    @subset :situation .== "5on5"
    _.name
end

players_position = @chain nhl_stats_1819 begin
    @subset :situation .== "5on5"
    _.position
end

points_per_game = @chain nhl_stats_1819 begin
    @subset :situation .== "5on5"
    @transform :points_per_game = :I_F_points ./ :games_played
    _.points_per_game
end

relative_corsi_percentage = @chain nhl_stats_1819 begin
    @subset :situation .== "5on5"
    @transform :relative_corsi_percentage = (:onIce_corsiPercentage .- :offIce_corsiPercentage) .* 100
    _.relative_corsi_percentage
end

games_played = @chain nhl_stats_1819 begin
    @subset :situation .== "5on5"
    _.games_played
end

# Appending labels
model_df = @chain model_df begin
    permutedims
    rename(:x1 => :UMAP1, :x2 => :UMAP2)
    @transform(
        :player_positions = players_position,
        :player_names = player_names,
        :points_per_game = points_per_game,
        :relative_corsi_percentage = relative_corsi_percentage,
        :games_played = games_played
    )
end

# Plotting umap
plot(
    model_df,
    x = :UMAP1,
    y = :UMAP2,
    color = players_position,
    Geom.point
)

# Interestingly, there are 3 distinct groups
# One comprised of defensemen, one of forwards
# And another of both, I wonder if this corresponds to points?

plot(
    model_df,
    x = :UMAP1,
    y = :UMAP2,
    color = :points_per_game,
    Geom.point
)

# It looks like there are a few outliers in these plots
plot(
    model_df,
    x = :UMAP1,
    y = :UMAP2,
    color = :relative_corsi_percentage,
    Geom.point
)

# This is the plot that explains it, the lower cluster are from players
# with low amounts of games played.
plot(
    model_df,
    x = :UMAP1,
    y = :UMAP2,
    color = :games_played,
    Geom.point
)

# For this UMAP, since I am testing the efficacy of advanced statistics
# Perhaps it would be pertinent in removing them prior to clustering.