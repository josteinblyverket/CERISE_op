#!/usr/bin/env python
# coding: utf-8

# In[55]:


import time
import os
import sys
import pyproj
import netCDF4
import datetime
import pyresample
import numpy as np
import matplotlib.pyplot as plt


# # Constants

# In[56]:


SGE_TASK_ID = 1
#
date_min = "20200901"
date_max = "20230602"
#
hours_AMSR2 = "H03"
#
paths = {}
paths["AMSR2"] = "/lustre/storeB/project/nwp/H2O/wp3/satellite_data/AMSR-2/"
paths["surfex"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR/archive/"
paths["surfex_grid"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR/climate/"
paths["output"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Training_data/"
#
surfex_inputgrid = paths["surfex_grid"] + "PGD.nc"
#
target_variables = ["tb6_h", "tb6_v", "tb7_h", "tb7_v", "tb10_h", "tb10_v", "tb18_h", "tb18_v", "tb23_h", "tb23_v", "tb36_h", "tb36_v"]
#
surfex_PGD_variables = ["ZS", "COVER004", "COVER006"]
surfex_prognostic_variables = ["FRAC_WATER", "FRAC_NATURE", "FRAC_SEA"]
surfex_surface_and_integrated_variables = ["Q2M_ISBA", "T2M_ISBA", "TS_ISBA", "DSN_T_ISBA", "LAI_ga", "PSN_ISBA", "PSNG_ISBA", "PSNV_ISBA"]
predictor_variables = surfex_prognostic_variables + surfex_surface_and_integrated_variables
#
n_soil_layers = 12
surfex_soil_variables = ["TG", "WSN_VEG", "WG", "WGI", "RSN_VEG", "SNOWTEMP", "SNOWLIQ", "HSN_VEG"] 
#
for var in surfex_soil_variables:
    for layer in range(1, n_soil_layers + 1):
        predictor_variables.append(var + str(layer) + "_ga")


# # Get surfex coordinates

# In[57]:


class get_surfex_coordinates():
    def __init__(self, surfex_inputgrid, surfex_PGD_variables):
        self.inputgrid = surfex_inputgrid
        self.surfex_PGD_variables = surfex_PGD_variables
    #
    def sfx2areadef(self, lat0, lon0, latori, lonori, xx, yy):
        proj2 = "+proj=lcc +lat_1=%.2f +lat_2=%.2f +lat_0=%.2f +lon_0=%.2f +units=m +ellps=WGS84 +no_defs" % (lat0,lat0,lat0,lon0)
        p2 = pyproj.Proj(proj2,preserve_units = False)
        origo = p2(lonori.data,latori.data)
        extent = origo + (origo[0] + xx[-1,-1], origo[1] + yy[-1,-1])
        area_def = pyresample.geometry.AreaDefinition("id2","hei2","lcc",proj2,xx.shape[1],yy.shape[0],extent)
        return(area_def)
    #
    def getSFXgrid(self):
        nc = netCDF4.Dataset(self.inputgrid, "r")
        lon0 = nc["LON0"][0]
        lat0 = nc["LAT0"][0]
        lonc = nc["LONORI"][0]
        latc = nc["LATORI"][0]
        #
        dx = nc["DX"][:]
        dy = nc["DY"][:]
        #
        xx = nc["XX"][:]
        yy = nc["YY"][:]
        nc.close()
        #
        x = xx[0,:]
        y = yy[:,0]
        #
        areadef = self.sfx2areadef(lat0, lon0, latc, lonc, xx, yy)
        return(areadef, x, y)
    #
    def load_PGD_variables(self):
        Surfex_PGD = {}
        nc = netCDF4.Dataset(self.inputgrid, "r")
        for var in self.surfex_PGD_variables:
            Surfex_PGD[var] = nc.variables[var][:,:]
        nc.close()
        return(Surfex_PGD)
    #
    def __call__(self):
        areadef, x, y = self.getSFXgrid()
        lon, lat = areadef.get_lonlats()
        Surfex_PGD = self.load_PGD_variables()
        #
        Surfex_coord = {}
        Surfex_coord["lon"] = lon
        Surfex_coord["lat"] = lat
        Surfex_coord["x"] = x
        Surfex_coord["y"] = y
        #
        return(Surfex_coord, Surfex_PGD)


# # List dates

# In[58]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Read AMSR2 data

# In[59]:


def read_AMSR2_data(date_task, paths, target_variables):
    AMSR2_data = {}
    #
    file_AMSR2 = paths["AMSR2"] + date_task[0:4] + "/" + date_task[4:6] + "/" + date_task[6:8] + "/AMSR2_SOR_TEST_" + date_task + ".nc"
    nc = netCDF4.Dataset(file_AMSR2, "r")
    dim_1D = nc.dimensions["N_DIM"].size
    dim_2D = (int(np.sqrt(dim_1D)), int(np.sqrt(dim_1D)))
    #
    for var in target_variables:
        var_data = np.flipud(np.reshape(nc.variables[var][:], dim_2D, order = "F"))
        var_data[var_data < 0] = np.nan
        AMSR2_data[var] = np.copy(var_data)
    #
    nc.close()
    return(AMSR2_data)


# # Read SURFEX data

# In[60]:


def read_surfex_data(date_task, paths, predictor_variables):
    Surfex_data = {}
    filename_constants = paths["surfex"] + "2023/01/01/00/SURFOUT.20230101_03h00.nc"
    previous_hours = "{:02d}".format(int(date_task[9:11]) - 3)
    path_task = paths["surfex"] + date_task[0:4] + "/" + date_task[4:6] + "/" + date_task[6:8] + "/" +  previous_hours + "/"
    print("SURFOUT." + date_task[0:8] + "_" + date_task[9:11] + "h00.nc")
    nc = netCDF4.Dataset(path_task + "SURFOUT." + date_task[0:8] + "_" + date_task[9:11] + "h00.nc", "r")
    #
    Surfex_coord = {}
    for var in ["XX", "YY"]:
        if var in nc.variables:
            var_data = nc.variables[var][:,:]
            if var == "XX":
                Surfex_coord["x"] = var_data[0,:]
            elif var == "YY":
                Surfex_coord["y"] = var_data[:,0]
        else:
            nc_constants = netCDF4.Dataset(filename_constants)
            var_data = nc_constants.variables[var][:,:]
            if var == "XX":
                Surfex_coord["x"] = var_data[0,:]
            elif var == "YY":
                Surfex_coord["y"] = var_data[:,0]
            nc_constants.close()
    #
    for var in ["PATCHP1", "PATCHP2"]:
        if var in nc.variables:
            var_data = nc.variables[var][:,:]
            var_data[var_data.mask == True] = 0
            Surfex_data[var] = np.expand_dims(var_data, axis = 0)
        else:
            nc_constants = netCDF4.Dataset(filename_constants)
            var_data = nc_constants.variables[var][:,:]
            var_data[var_data.mask == True] = 0
            Surfex_data[var] = np.expand_dims(var_data, axis = 0)
            nc_constants.close()
    #
    for var in predictor_variables:
        try:
            if (var in nc.variables) or (var.replace("_ga", "P1") in nc.variables):
                if "_ga" in var:
                    var_data_P1 = nc.variables[var.replace("_ga", "P1")][:,:]
                    var_data_P2 = nc.variables[var.replace("_ga", "P2")][:,:]
                    var_data_P1 = np.expand_dims(var_data_P1, axis = 0)
                    var_data_P2 = np.expand_dims(var_data_P2, axis = 0)
                    Surfex_data[var] = np.nansum([Surfex_data["PATCHP1"] * var_data_P1, Surfex_data["PATCHP2"] * var_data_P2], axis = 0)
                    Surfex_data[var][var_data_P1.mask == True] = var_data_P2[var_data_P1.mask == True]
                    Surfex_data[var][var_data_P2.mask == True] = var_data_P1[var_data_P2.mask == True]
                    Surfex_data[var][np.logical_and(var_data_P1.mask == True, var_data_P2.mask == True)] = np.nan
                else:
                    Surfex_data[var] = nc.variables[var][:,:]
            else:
                #print("prognostic variable: ", var)
                ncp = netCDF4.Dataset(path_task + "SURFOUT.nc")
                if "_ga" in var:
                    var_data_P1 = ncp.variables[var.replace("_ga", "P1")][:,:]
                    var_data_P2 = ncp.variables[var.replace("_ga", "P2")][:,:]
                    var_data_P1 = np.expand_dims(var_data_P1, axis = 0)
                    var_data_P2 = np.expand_dims(var_data_P2, axis = 0)
                    Surfex_data[var] = np.nansum([Surfex_data["PATCHP1"] * var_data_P1, Surfex_data["PATCHP2"] * var_data_P2], axis = 0)
                    Surfex_data[var][var_data_P1.mask == True] = var_data_P2[var_data_P1.mask == True]
                    Surfex_data[var][var_data_P2.mask == True] = var_data_P1[var_data_P2.mask == True]
                    Surfex_data[var][np.logical_and(var_data_P1.mask == True, var_data_P2.mask == True)] = np.nan
                else:
                    Surfex_data[var] = ncp.variables[var][:,:]
                ncp.close()
        except:
            print("Variable not found: " + var)
            if (var == "SNOWTEMP9_ga") or (var == "SNOWLIQ9_ga"):
                pass
            else:
                sys.exit()
    #
    nc.close()
    return(Surfex_data)


# In[61]:


def calculate_WSN_T_ISBA(Surfex_data, n_soil_layers):
    WSN_T_ISBA = np.zeros(np.shape(Surfex_data["WSN_VEG1_ga"]))
    for layer in range(0, n_soil_layers):
        WSN_T_ISBA = WSN_T_ISBA + Surfex_data["WSN_VEG" + str(layer + 1) + "_ga"]
    return(WSN_T_ISBA)


# # Write netCDF output

# In[62]:


def write_netcdf(date_task, paths, Surfex_coord, Surfex_PGD, Targets, Surfex_data):
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/"
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "Dataset_" + date_task + ".nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    output_netcdf = netCDF4.Dataset(output_filename, 'w', format = 'NETCDF4')
    #
    x = output_netcdf.createDimension("x", len(Surfex_coord["x"]))
    y = output_netcdf.createDimension("y", len(Surfex_coord["y"]))
    #
    Outputs = vars()
    #
    for var in Targets:
        Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
        Outputs[var][:,:] = Targets[var]
    for var in Surfex_PGD:
        Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
        Outputs[var][:,:] = Surfex_PGD[var]        
    for var in Surfex_data:
        Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
        Outputs[var][:,:] = np.squeeze(Surfex_data[var])
    #
    output_netcdf.close() 


# # Data processing 

# In[63]:


list_dates = make_list_dates(date_min, date_max)
date_task = list_dates[SGE_TASK_ID - 1] + hours_AMSR2
#
Surfex_coord, Surfex_PGD = get_surfex_coordinates(surfex_inputgrid, surfex_PGD_variables)()
Targets = read_AMSR2_data(date_task, paths, target_variables)
Surfex_data = read_surfex_data(date_task, paths, predictor_variables)
Surfex_data["WSN_T_ISBA"] = calculate_WSN_T_ISBA(Surfex_data, n_soil_layers)
Surfex_data["FRAC_LAND_AND_SEA_WATER"] = Surfex_data["FRAC_WATER"] + Surfex_data["FRAC_SEA"] 
Surfex_data["SNOW_GRADIENT"] = (Surfex_data["SNOWTEMP12_ga"] - Surfex_data["SNOWTEMP1_ga"]) / Surfex_data["DSN_T_ISBA"]
Surfex_data["SNOW_GRADIENT"][Surfex_data["SNOW_GRADIENT"] > 50] = 50
Surfex_data["SNOW_GRADIENT"][Surfex_data["SNOW_GRADIENT"] < -50] = -50
#
write_netcdf(date_task, paths, Surfex_coord, Surfex_PGD, Targets, Surfex_data)

