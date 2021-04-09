# this program download GOES data using the toolbox GOES available on github (with quite detailed and useful guide): https://github.com/joaohenry23/GOES

import GOES

path = '/home/j/Desktop/PROJ_WORK_Thesis/GOES_data/' # change this path

#download GOES data for the CLOUD mask for 2 Feb 2020 between 4 and 6 am in UTC time
GOES.download('goes16', 'ABI-L2-ACMF',
             DateTimeIni = '20200202-040000', DateTimeFin = '20200202-060400',  path_out=path)

# get a list of all files
flist = GOES.locate_files(path, 'OR_ABI-L2-CMIPF-M*C02_G16*.nc',
                          '20200202-120000', '20200202-210100')
