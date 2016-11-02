import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
servo = pd.read_csv('Datasets/servo.data',
                    names=['motor', 'screw', 'pgain', 'vgain', 'class'])


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the
# length of (# of samples in) that slice:
#
print('Numer of samples where vgain equals 5: {}'.format(
                                           len(servo.ix[servo.vgain == 5, :])))


# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
print('Number of samples where motor equals E and screw equals E: {}'.format(
                len(servo.ix[(servo.motor == 'E') & (servo.screw == 'E'), :])))


# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
print('vgain mean when pgain equals 4: {}'.format(
                                      servo.vgain.ix[servo.pgain == 4].mean()))


# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!

print('Servo dataframe dtypes')
print(servo.dtypes)
