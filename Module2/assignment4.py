import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#

url = 'http://www.espn.com/nhl/statistics/player/_/stat/points/\
sort/points/year/2015/seasontype/2'
table = pd.read_html(url, skiprows=0, header=1)


# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
df = table[0].copy()
df.columns = ['RK', 'Player', 'Team', 'Games_Played', 'Goals', 'Assists',
              'Points', 'Plus_Minus_Rating', 'Penalty_Minutes',
              'Points_Per_Game', 'Shots_on_Goal', 'Shooting_Percentage',
              'Game_Winning_Goals', 'Power_Play_Goals', 'Power_Play_Assists',
              'Short_Handed_Goals', 'Short_Handed_Assists']


# TODO: Get rid of any row that has at least 4 NANs in it
#
df = df.dropna(axis=0, thresh=(len(df.columns)-3))

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# I'd use df.ix[df.RK != 'RK', :]
removable_indexes = df.ix[df.RK == 'RK', :].index
df.drop(labels=removable_indexes, axis=0, inplace=True)


# TODO: Get rid of the 'RK' column
#
df.drop(labels='RK', axis=1, inplace=True)


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df.reset_index(drop=True, inplace=True)


# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
numeric_columns = list(df.columns[2:])
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric,
                                                args=('coerce',))


# TODO: Your dataframe is now ready! Use the appropriate
# commands to answer the questions on the course lab page.
print('The dataframe has {} rows,'.format(len(df)))
print('{} unique PCT values in the table'.format(len(
                                      df.Shooting_Percentage.unique())))
print('and we get {} by adding GP[15] and GP[16]'.format(
      df.Games_Played[15] + df.Games_Played[16]))
