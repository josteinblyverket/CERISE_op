import pyproj
from pyproj import Proj
import pyresample
import netCDF4 as nc


def sfx2areadef(lat0,lon0,latori,lonori,xx,yy):

    print(latori.data)

    proj2 = "+proj=lcc +lat_1=%.2f +lat_2=%.2f +lat_0=%.2f +lon_0=%.2f +units=m +ellps=WGS84 +no_defs" % (lat0,lat0,lat0,lon0)
    p2 = pyproj.Proj(proj2,preserve_units=False)
    origo = p2(lonori.data,latori.data)
    print(origo)
    extent = origo + (origo[0] + xx[-1,-1], origo[1]+yy[-1,-1])
    area_def = pyresample.geometry.AreaDefinition("id2","hei2","lcc",proj2,xx.shape[1],yy.shape[0],extent)

    return area_def

def getSFXgrid():

    pgd = nc.Dataset('/ec/res4/scratch/sbjb/sfx_data/CERISE_Land_Pv2_dev02/climate/PGD.nc','r')    
    lon0 = pgd["LON0"][0]
    lat0 = pgd["LAT0"][0]
    lonc = pgd["LONORI"][0]
    latc = pgd["LATORI"][0]

    dx = pgd["DX"][:]
    dy = pgd["DY"][:]

    xx = pgd["XX"][:]
    yy = pgd["YY"][:]
    areadef =  sfx2areadef(lat0,lon0,latc,lonc,xx,yy)

    return areadef, xx, yy
