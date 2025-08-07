import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import shutil
import os
import dask

def write_existing_forcing(values, filename, cfg):
    """ write forcing
    values: array (nvar, ntimes, npoints)
    """
    with nc.Dataset(filename, "r+") as f:
        for i in range(len(cfg["variables"])):
            f.variables[cfg["variables"][i]["name"]][:] = values[i][:]


def write_new_forcing(values, filename, time, Number_of_points, dt, cfg):
    nvar = values.shape[0]
    met_path = "EnKF forcing noise"

    with nc.Dataset(filename, 'w', format='NETCDF4') as f:

        # Dim the dimensions of NetCDF
        f.createDimension('time', time)
        f.createDimension('Number_of_points', Number_of_points)

        FRC_STP = f.createVariable('FRC_TIME_STP', 'f4')
        FRC_STP.units = "s"
        FRC_STP[:] = dt  # Forcing timestep
        TIME = f.createVariable('time', 'f4', ('time',))
        TIME.units = "hours"
        time = range(0, time, 1)
        TIME[:] = time[:]
        variables = []
        for i in range(nvar):
            #print(i, cfg["variables"][i])
            variables += [f.createVariable(cfg["variables"][i]["name"], 'f4', ('time', 'Number_of_points'), fill_value=1.0e+20)]
            variables[i].missing_value = 1.0e+20
            variables[i].long_name = cfg["variables"][i]["name"]
            variables[i].history = met_path
            variables[i].coordinates = "Number_of_points"
            variables[i][:, :] = values[i, :, :]


def read_forcing(filename, cfg, times=None):
    """ Read forcing variables
    returns: array (nvar, ntimes, npoints)
    """
    if times is not None:
        tslice = times
    else:
        tslice = slice(None, None)
    varlist = []
    with nc.Dataset(filename) as f:
        for i in range(len(cfg["variables"])):
            #print(cfg["variables"][i]["name"])
            if cfg["variables"][i]["name"] == "ZS":
                varlist += [f.variables[cfg["variables"][i]["name"]][:]]
            else:
                varlist += [f.variables[cfg["variables"][i]["name"]][tslice, :]]

    return np.stack(varlist)


def read_state(filename, varnames=None, idx=None):
    """
    return array with dimenstions (..., vars)
    """
    sle = []
    with nc.Dataset(filename) as f:
        for j in range(len(varnames)):
            sle += [f.variables[varnames[j]][:]]
    return ma.masked_invalid(ma.masked_values(ma.stack(sle, axis=-1), 1e20))


def read_state_1(filename, varnames=None, idx=None):
    """
    return array with dimenstions (..., vars)
    """
    with nc.Dataset(filename) as f:
        for j in range(len(varnames)):
            tmp = f.variables[varnames[j]][:]
            if j == 0:
                sle = ma.zeros((len(varnames),) + tmp.shape)
            sle[j] = tmp
    return np.moveaxis(sle, 0, -1)


def read_state_2(filename, varnames=None, idx=None, a=None):
    """
    return array with dimenstions (..., vars)
    """
    inplace = False
    if a is not None:
        if a.shape[0] == len(varnames):
            inplace = True
    with nc.Dataset(filename) as f:
        #print("open: ", filename)
        if inplace:
            for j in range(len(varnames)):
                a[j] = f.variables[varnames[j]][:]
        else:        
            for j in range(len(varnames)):
                tmp = f.variables[varnames[j]][:]
                if j == 0:
                    sle = np.zeros((len(varnames),) + tmp.shape)
                sle[j] = tmp
            return sle


def read_forcing_ens(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        sl += [np.moveaxis(read_forcing(fname, **kwargs),0,1)]
    return ma.stack(sl, axis=-1)

def read_ens(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        sl += [read_state(fname, **kwargs)]
    return ma.stack(sl, axis=-1)

def read_ens_1(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    #sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        tmp = read_state(fname, **kwargs)
        if i == 0:
            sl = ma.zeros((nens,) + tmp.shape)
        sl[i] = tmp
    return np.moveaxis(sl, 0, -1)


def read_ens_2(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    #sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        tmp = read_state_1(fname, **kwargs)
        if i == 0:
            sl = ma.zeros((nens,) + tmp.shape)
        sl[i] = tmp
    return np.moveaxis(sl, 0, -1)


def read_ens_3(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        if i == 0:
            tmp = read_state_2(fname, **kwargs)
            sl = np.zeros((nens,) + tmp.shape)
            sl[0] = tmp
        else:
            read_state_2(fname, a=sl[i], **kwargs)
    return ma.masked_values(np.moveaxis(sl, (0, 1), (-1, -2)), 1e20)


def read_ens_4(file_pattern, nens=1, outarr=None, **kwargs):
    """
    return array with dimension (..., nens)
    """
    sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        sl.append(dask.delayed(read_state_2)(fname, a=outarr[i], **kwargs))
    out = dask.compute(*sl)
    #return np.moveaxis(np.stack(out), (0, 1), (-1, -2))



def read_ens_5(file_pattern, nens=1, **kwargs):
    """
    return array with dimension (..., nens)
    """
    sl = []
    for i in range(nens):
        if "@mbr@" in file_pattern:
            fname = file_pattern.replace("@mbr@", "%03d" % i)
        else:
            fname = file_pattern
        sl.append(dask.delayed(read_state_2)(fname, **kwargs))
    out = dask.compute(*sl)
    return np.moveaxis(np.stack(out), (0, 1), (-1, -2))



def write_existing_state(filename, varnames=None, values=None, vardim=0):
    """
    write values (..., vars) to file
    """
    with nc.Dataset(filename, 'r+') as f:
        for i in range(len(varnames)):
            f.variables[varnames[i]][:] = values[..., i]
#            print("file", f.variables[varnames[i]][:].shape)
#            print("vals", values[..., i].shape)


def write_new_state(filename, varnames, values, vardim=0):
    """  """

    with nc.Dataset(filename, 'w', format='NETCDF4') as f:
        f.createDimension('xx', values.shape[2])
        f.createDimension('yy', values.shape[1])
        f.createDimension('Number_of_Patches', values.shape[0])
        variables = []
        #ind = [slice(None,None),]*len(values.shape)
        print(values.shape)
        for i in range(len(varnames)):
          #  ind[vardim] = i
         #   print(ind)
            variables += [f.createVariable(varnames[i], 'f4', ('Number_of_Patches', "yy", "xx"), fill_value=1.0e+20)]
            variables[i].missing_value = 1.0e+20
            variables[i].long_name = varnames[i]
            variables[i].history = "SFX state"
            variables[i].coordinates = "y, x"
            variables[i][:, :, :] = values[:, :, :, i]


def write_ens(file_pattern, output_pattern, values, nens=1, **kwargs):
    """ write values (..., ens)  to files """
    for i in range(nens):
        fnam_in = file_pattern.replace("@mbr@", "%03d" % i)
        fnam_out = output_pattern.replace("@mbr@", "%03d" % i)
        os.makedirs(os.path.dirname(fnam_out), exist_ok=True)
        shutil.copyfile(fnam_in, fnam_out)
        write_existing_state(fnam_out, values=values[..., i], **kwargs)


def read_n_cache(opath, tpath, varnames, nens=1, force=False, debug=True):
    """ 
    opath: path to fa/ncfile
    tpath: path to nemporary npy file
    """
    #print("opath", opath)
    #print("tpath", tpath)
    if os.path.isfile(tpath) and not force:
        if debug:
            print("fast read")
        return ma.masked_values(np.load(tpath), 1e20)
    else:
        if debug:
            print("slow read")
        values = read_ens(opath, nens=nens, varnames=varnames)
        np.save(tpath, values.filled(1e20))
        return values


def fill_pattern(pattern, dt=None, mbr=None, exp=None):
    pout = pattern
    if dt is not None:
        pout = pout.replace("@yyyy@", dt.strftime("%Y"))
        pout = pout.replace("@mm@", dt.strftime("%m"))
        pout = pout.replace("@dd@", dt.strftime("%d"))
        pout = pout.replace("@hh@", dt.strftime("%H"))
    if mbr is not None:
        if type(mbr) is str:
            if len(mbr) > 0:
                mbrstr = mbr
            else:
                mbrstr = ""
        elif type(mbr) is int:
            mbrstr = "%03d" % mbr
        else:
            raise NotImplementedError
        pout = pout.replace("@mbr@", mbrstr)
    if exp is not None:
        pout = pout.replace("@exp@", exp)
    return pout