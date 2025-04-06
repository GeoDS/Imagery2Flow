import os
from osgeo import gdal
import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_tif_path', type=str)
parser.add_argument('--output_tif_dir', type=str, default='../data/')
parser.add_argument('--region', type=str, default='M1') 
parser.add_argument('--year', type=str, default='2020')
parser.add_argument('--tag', type=str, default='l8') # s2 or l8
parser.add_argument('--shp_path', type=str) # spatial unit 

if __name__ == '__main__':
    args = parser.parse_args()

    shps = gpd.read_file(args.shps_path)
    for _, rows in shps.iterrows():
        data_path = os.path.join(args.output_tif_dir, args.region, args.tag + "_" + args.year)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        tif_out_name = args.tag+"_"+args.year+"_"+rows['GEOID20']+".tif"
        tif_out = os.path.join(data_path, tif_out_name)

        ds = gdal.Warp(tif_out, args.input_tif_path,
                        format='GTiff', cutlineDSName=rows['geometry'], cropToCutline=True, dstNodata=0)
        del ds

