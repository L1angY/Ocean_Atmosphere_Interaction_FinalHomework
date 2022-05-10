"""
本代码主要利用An and Jin(2004)中的混合层温度倾向方程来分析ENSO过程
本代码先简单写出一个雏形,并简单看一下1997~1998年9项之和从97年4月到98年6月的变化

基本的思路其实就是利用np.gradient来实现中央差分(边缘是前后差),
需要注意的问题有两点:
①注意量纲分析,每一项的单位都是℃/mon,所以中央差的结果需要乘上30*24*60*60
②需要将经纬度间隔转换为m,即每度平均是111km
"""

# %%
import numpy as np
import xarray as xr
import proplot as plot

def read_SODA_UVWT_Upper55m(path, DownResolution=True):
    """
    path: 文件路径

    Return:
    lat: 纬度
    lon: 经度
    depth: 深度

    u_12mons_8020_ltm:气候态的12个月的u         shape = (mon, depth, lat, lon)
    v_12mons_8020_ltm:气候态的12个月的v         shape = (mon, depth, lat, lon)
    w_12mons_8020_ltm:气候态的12个月的w         shape = (mon, depth, lat, lon)
    temp_12mons_8020_ltm:气候态的12个月的temp   shape = (mon, depth, lat, lon)

    u_12mon_8020:每年12个月的u          shape = (year, mon, depth, lat, lon)
    v_12mon_8020:每年12个月的v          shape = (year, mon, depth, lat, lon)
    w_12mon_8020:每年12个月的w          shape = (year, mon, depth, lat, lon)
    temp_12mon_8020:每年12个月的温度    shape = (year, mon, depth, lat, lon)
    """

    data = xr.open_dataset(path)
    depth = data.st_ocean.data

    if DownResolution:
        lat = data.lat.data[::2]
        lon = data.lon.data[::2]
        u = data['u'].data[:, :, ::2, ::2]
        v = data['v'].data[:, :, ::2, ::2]
        w = data['wt'].data[:, :, ::2, ::2]
        temp = data['temp'].data[:, :, ::2, ::2]
    else:
        lat = data.lat.data
        lon = data.lon.data
        u = data['u'].data
        v = data['v'].data
        w = data['wt'].data
        temp = data['temp'].data

    # 取出1980-2020年12个月的场
    u_12mons_8020 = np.empty((41, 12, len(depth), len(lat), len(lon)))
    v_12mons_8020 = np.empty((41, 12, len(depth), len(lat), len(lon)))
    w_12mons_8020 = np.empty((41, 12, len(depth), len(lat), len(lon)))
    temp_12mons_8020 = np.empty((41, 12, len(depth), len(lat), len(lon)))

    for i in range(41):
        index = 12*i
        u_12mons_8020[i, :] = u[index:index+12]
        v_12mons_8020[i, :] = v[index:index+12]
        w_12mons_8020[i, :] = w[index:index+12]
        temp_12mons_8020[i, :] = temp[index:index+12]

    u_12mons_8020_ltm = u_12mons_8020.mean(axis=0)
    v_12mons_8020_ltm = v_12mons_8020.mean(axis=0)
    w_12mons_8020_ltm = w_12mons_8020.mean(axis=0)
    temp_12mons_8020_ltm = temp_12mons_8020.mean(axis=0)

    # 扣除线性趋势, 并顺便扣除季节平均
    for mon in range(12):
        for Zlevel in range(len(depth)):
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if (np.isnan(u_12mons_8020[:, mon, Zlevel, i, j].sum())):  # 排掉陆地值点
                        u_12mons_8020[:, mon, Zlevel, i, j] = np.nan
                    else:
                        x = np.arange(1, 42, 1)
                        z = np.polyfit(x, u_12mons_8020[:, mon, Zlevel, i, j], 1)
                        y = np.polyval(z, x)

                        u_12mons_8020[:, mon, Zlevel, i, j] = \
                            u_12mons_8020[:, mon, Zlevel, i, j] - y

                    if (np.isnan(v_12mons_8020[:, mon, Zlevel, i, j].sum())):  # 排掉陆地值点
                        v_12mons_8020[:, mon, Zlevel, i, j] = np.nan
                    else:
                        x = np.arange(1, 42, 1)
                        z = np.polyfit(x, v_12mons_8020[:, mon, Zlevel, i, j], 1)
                        y = np.polyval(z, x)

                        v_12mons_8020[:, mon, Zlevel, i, j] = \
                            v_12mons_8020[:, mon, Zlevel, i, j] - y

                    if (np.isnan(w_12mons_8020[:, mon, Zlevel, i, j].sum())):  # 排掉陆地值点
                        w_12mons_8020[:, mon, Zlevel, i, j] = np.nan
                    else:
                        x = np.arange(1, 42, 1)
                        z = np.polyfit(x, w_12mons_8020[:, mon, Zlevel, i, j], 1)
                        y = np.polyval(z, x)

                        w_12mons_8020[:, mon, Zlevel, i, j] = \
                            w_12mons_8020[:, mon, Zlevel, i, j] - y

                    if (np.isnan(temp_12mons_8020[:, mon, Zlevel, i, j].sum())):  # 排掉陆地值点
                        temp_12mons_8020[:, mon, Zlevel, i, j] = np.nan
                    else:
                        x = np.arange(1, 42, 1)
                        z = np.polyfit(x, temp_12mons_8020[:, mon, Zlevel, i, j], 1)
                        y = np.polyval(z, x)

                        temp_12mons_8020[:, mon, Zlevel, i, j] = \
                            temp_12mons_8020[:, mon, Zlevel, i, j] - y

    return lat, lon, depth, \
           u_12mons_8020_ltm, v_12mons_8020_ltm, w_12mons_8020_ltm, temp_12mons_8020_ltm, \
           u_12mons_8020, v_12mons_8020, w_12mons_8020, temp_12mons_8020


def get_differentiate_10terms(depth, lat, lon, latlon_resolution,
                              u, v, w, temp, temp_lastmonth,  # temp_lastmonth是上一个月数据
                              u_ltm, v_ltm, w_ltm, temp_ltm):
    """
    对每个月的数据进行10项差分

    input:
        depth: 深度数组
        lat: 纬度数组
        lon: 经度数组
        latlon_resolution: 经纬度分辨率(unit: degree), (由于只是分析热带地区,直接默认1°=111km)

        u: u的距平
        v: v的距平
        w: w的距平
        temp: 温度的距平
        temp_lastmonth: 上一个月的温度的距平
        u_ltm: 对应月份的u气候态
        v_ltm: 对应月份的v气候态
        w_ltm: 对应月份的w气候态
        temp_ltm: 对应月份的温度气候态
    return:
        LeftOneTerms: 方程左边那一项, shape = (depth, lat, lon)
        RightNineTerms: 方程右边那九项, shape = (term, depth, lat, lon)
        (最后需要将NineTerms结构乘于30*24*60*60=2,592,000秒,才能和左边的℃/mon对应)
    """

    LeftOneTerms = temp - temp_lastmonth

    RightNineTerms = np.empty((9, len(depth), len(lat), len(lon)))


    RightNineTerms[0] = (-1) * u * np.gradient(temp_ltm, 111 * 1000 * latlon_resolution,
                                          axis=2)
    RightNineTerms[1] = (-1) * v * np.gradient(temp_ltm, 111 * 1000 * latlon_resolution,
                                          axis=1)
    RightNineTerms[2] = (-1) * w * np.gradient(temp_ltm, -depth, axis=0)

    # --------------------------------------

    RightNineTerms[3] = (-1) * u_ltm * np.gradient(temp, 111 * 1000 * latlon_resolution,
                                              axis=2)
    RightNineTerms[4] = (-1) * v_ltm * np.gradient(temp, 111 * 1000 * latlon_resolution,
                                              axis=1)
    RightNineTerms[5] = (-1) * w_ltm * np.gradient(temp, -depth, axis=0)

    # --------------------------------------

    RightNineTerms[6] = (-1) * u * np.gradient(temp, 111 * 1000 * latlon_resolution,
                                          axis=2)
    RightNineTerms[7] = (-1) * v * np.gradient(temp, 111 * 1000 * latlon_resolution,
                                          axis=1)
    RightNineTerms[8] = (-1) * w * np.gradient(temp, -depth, axis=0)

    RightNineTerms = RightNineTerms * (30 * 24 * 60 * 60)

    return LeftOneTerms, RightNineTerms


# %%
lat, lon, depth, \
u_12mons_8020_ltm, v_12mons_8020_ltm, w_12mons_8020_ltm, temp_12mons_8020_ltm, \
u_12mons_8020, v_12mons_8020, w_12mons_8020, temp_12mons_8020 = \
    read_SODA_UVWT_Upper55m(
        r'E:\SODA3.15.2_mn_ocean\soda3.15.2_mn_ocean_reg_1980-2020_select_remap_TroPacific.nc',)

# %%
u = u_12mons_8020[1997-1980, 7]
v = v_12mons_8020[1997-1980, 7]
w = w_12mons_8020[1997-1980, 7]
temp = temp_12mons_8020[1997-1980, 7]
temp_lastmonth = temp_12mons_8020[1997-1980, 6]
u_ltm = u_12mons_8020_ltm[7]
v_ltm = v_12mons_8020_ltm[7]
w_ltm = w_12mons_8020_ltm[7]
temp_ltm = temp_12mons_8020_ltm[7]



LeftOneTerm, RightNineTerms = get_differentiate_10terms(
                                      depth, lat, lon, lat[1] - lat[0],
                                      u, v, w, temp, temp_lastmonth,
                                      u_ltm, v_ltm, w_ltm, temp_ltm)

LeftOneTerm_Zmean = np.mean(LeftOneTerm, axis=0)
RightNineTerms_Zmean = np.mean(RightNineTerms, axis=1)

# %% 绘图2
subplot_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 0]]
plot.rc.reso = 'hi'  # 海岸线使用分辨率 'hi' 'med' 'lo'
proj = plot.Proj('cyl', lon_0=lon.mean())
fig = plot.figure(space=4, axwidth=4, sharey=False)
axs = fig.subplots(subplot_array, proj=proj)
axs.format(
    abc='(a)', abcloc='ul', abcsize=13, gridlabelsize=10, gridlabelweight='heavy',
    labels=True, lonlines=30, latlines=5, latlim=(-10.1, 10.1),
    lonlim=(lon.min()-0.1, lon.max()), gridminor=True, coast=True, coastlinewidth=1,
    coastcolor='gray', titlesize=15, titleweight='heavy',
    suptitle='1997 August', suptitlesize=16)


ulname = [r'$-u^{\prime} \partial x \bar{T}$', r'$-v^{\prime} \partial y \bar{T}$', r'$-w^{\prime} \partial z \bar{T}$',
          r'$-\bar{u} \partial x T^{\prime}$', r'$-\bar{v} \partial y T^{\prime}$', r'$-\bar{w} \partial z T^{\prime}$',
          r'$-u^{\prime} \partial x T^{\prime}$', r'$-v^{\prime} \partial y T^{\prime}$', r'$-w^{\prime} \partial z T^{\prime}$',
          r'Sum of nine advection terms', r'$\partial T^{\prime} / \partial t$']
cmap = plot.Colormap('NegPos')

for i in range(9):
    axs[i].format(ltitle=ulname[i])
    m = axs[i].contourf(lon, lat, RightNineTerms_Zmean[i],
                        cmap=cmap, cmap_kw={'cut': -0.1},
                        levels=np.arange(-1.6, 1.61, 0.2),
                        extend='both')

axs[9].format(ltitle=ulname[9], titlesize=11)
m = axs[9].contourf(lon, lat, RightNineTerms_Zmean.sum(axis=0),
                    cmap=cmap, cmap_kw={'cut': -0.1},
                    levels=np.arange(-1.6, 1.61, 0.2),
                    extend='both')

axs[10].format(ltitle=ulname[10])
m = axs[10].contourf(lon, lat, LeftOneTerm_Zmean,
                    cmap=cmap, cmap_kw={'cut': -0.1},
                    levels=np.arange(-1.6, 1.61, 0.2),
                    extend='both')

fig.colorbar(m, loc='b', width=0.12, length=0.9, ticklabelsize=12,
             extendsize='3em', tickdir='in')

fig.save(r'C:\Users\59799\Desktop\test.png',
         dpi=600)
plot.close()  # figure的GUI界面经常卡死(图片像素太高),所以直接保存后直接关了
