"""
本代码主要利用An and Jin(2004)中的混合层温度倾向方程来分析ENSO过程

本代码的目标是:分析La Nina再发展与否的动力过程分析
La Nina持续年:1984,1999,2000,2008,2011(为了对比强烈一些,衰退为中性的没有考虑)
La Nina转El Nino:2006,2009,2018

由于不知道哪一项比较重要,所以先把所有项画出来,然后再看看哪一项比较重要
结果的呈现主要利用霍夫莫勒图(即横轴经度,纵轴是时间,3°S~3°N平均)
检验就算了,懒得弄了
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


# %% 读取数据
lat, lon, depth, \
u_12mons_8020_ltm, v_12mons_8020_ltm, w_12mons_8020_ltm, temp_12mons_8020_ltm, \
u_12mons_8020, v_12mons_8020, w_12mons_8020, temp_12mons_8020 = \
    read_SODA_UVWT_Upper55m(
    r'E:\SODA3.15.2_mn_ocean\soda3.15.2_mn_ocean_reg_1980-2020_select_remap_TroPacific.nc',
    DownResolution=True)


# %% 接下来就是要按照两个分类的年份取出数据并进行差分(每一个月差分一次)
years_Per = np.array([1984,1999,2000,2008,2011])
years_Trans = np.array([2006,2009,2018])

LeftOneTerm_Per = np.empty((12, len(lat), len(lon))) #直接存储垂直方向平均的,避免一堆变量
RightNineTerms_Per = np.empty((9, 12, len(lat), len(lon)))

LeftOneTerm_Trans = np.empty_like(LeftOneTerm_Per)
RightNineTerms_Trans = np.empty_like(RightNineTerms_Per)

for mon in range(12):
    u = u_12mons_8020[years_Per-1980, mon].mean(axis=0)
    v = v_12mons_8020[years_Per-1980, mon].mean(axis=0)
    w = w_12mons_8020[years_Per-1980, mon].mean(axis=0)
    temp = temp_12mons_8020[years_Per-1980, mon].mean(axis=0)

    if (mon == 0):
        temp_lastmonth = temp_12mons_8020[years_Per-1980-1, -1].mean(axis=0)
        u_ltm = u_12mons_8020_ltm[-1]
        v_ltm = v_12mons_8020_ltm[-1]
        w_ltm = w_12mons_8020_ltm[-1]
        temp_ltm = temp_12mons_8020_ltm[-1]
    else:
        temp_lastmonth = temp_12mons_8020[years_Per-1980, mon-1].mean(axis=0)
        u_ltm = u_12mons_8020_ltm[mon-1]
        v_ltm = v_12mons_8020_ltm[mon-1]
        w_ltm = w_12mons_8020_ltm[mon-1]
        temp_ltm = temp_12mons_8020_ltm[mon-1]

    a, b = get_differentiate_10terms(depth, lat, lon, lat[1] - lat[0],
                                     u, v, w, temp, temp_lastmonth,
                                     u_ltm, v_ltm, w_ltm, temp_ltm)

    LeftOneTerm_Per[mon], RightNineTerms_Per[:, mon] = \
        a.mean(axis=0), b.mean(axis=1)
    # -----------------------------------------------------------------

    u = u_12mons_8020[years_Trans - 1980, mon].mean(axis=0)
    v = v_12mons_8020[years_Trans - 1980, mon].mean(axis=0)
    w = w_12mons_8020[years_Trans - 1980, mon].mean(axis=0)
    temp = temp_12mons_8020[years_Trans - 1980, mon].mean(axis=0)

    if (mon == 0):
        temp_lastmonth = temp_12mons_8020[years_Trans-1980-1, -1].mean(axis=0)
        u_ltm = u_12mons_8020_ltm[-1]
        v_ltm = v_12mons_8020_ltm[-1]
        w_ltm = w_12mons_8020_ltm[-1]
        temp_ltm = temp_12mons_8020_ltm[-1]
    else:
        temp_lastmonth = temp_12mons_8020[years_Trans-1980, mon-1].mean(axis=0)
        u_ltm = u_12mons_8020_ltm[mon-1]
        v_ltm = v_12mons_8020_ltm[mon-1]
        w_ltm = w_12mons_8020_ltm[mon-1]
        temp_ltm = temp_12mons_8020_ltm[mon-1]

    a, b = get_differentiate_10terms(depth, lat, lon, lat[1] - lat[0],
                                     u, v, w, temp, temp_lastmonth,
                                     u_ltm, v_ltm, w_ltm, temp_ltm)

    LeftOneTerm_Trans[mon], RightNineTerms_Trans[:, mon] = \
        a.mean(axis=0), b.mean(axis=1)


# 最后就是纬度平均一下(3°S~3°N)
LeftOneTerm_Per_Latmean = LeftOneTerm_Per[:, 14:28].mean(axis=1)
RightNineTerms_Per_Latmean = RightNineTerms_Per[:, :, 14:28].mean(axis=2)

LeftOneTerm_Trans_Latmean = LeftOneTerm_Trans[:, 14:28].mean(axis=1)
RightNineTerms_Trans_Latmean = RightNineTerms_Trans[:, :, 14:28].mean(axis=2)

# %% 绘图
subplot_array = [[1, 10], [2, 11], [3, 12], 
                 [4, 13], [5, 14], [6, 15],
                 [7, 16], [8, 17], [9, 18]]
fig = plot.figure(space=4, axwidth=4, 
                  sharey=False, sharex=False, refaspect=3.5)
axs = fig.subplots(subplot_array)

axs.format(abc='a)', abcloc='ul', abcsize=20,
           ytickminor=False, xlim=(160, 270), 
           yticklabels=['Jan', 'Mar',  'May', 
                        'Jul', 'Sep', 'Nov', ],
           ticklabelsize=10, ylim=(-12, -1),
           xticks=np.arange(160, 271, 20), yticks=np.arange(-12, 0, 2),
           xticklabels=['160°E', '180', '160°W', '140°W', '120°W', '100°W'],
           toplabels=('La Niña Persistence', 'La Niña to El Niño'),
           leftlabels=(r'$-u^{\prime} \partial x \bar{T}$', 
                       r'$-v^{\prime} \partial y \bar{T}$', 
                       r'$-w^{\prime} \partial z \bar{T}$',
                       r'$-\bar{u} \partial x T^{\prime}$', 
                       r'$-\bar{v} \partial y T^{\prime}$', 
                       r'$-\bar{w} \partial z T^{\prime}$',
                       r'$-u^{\prime} \partial x T^{\prime}$', 
                       r'$-v^{\prime} \partial y T^{\prime}$', 
                       r'$-w^{\prime} \partial z T^{\prime}$'),
           toplabelsize=16, leftlabelsize=16)

cmap = plot.Colormap('NegPos')

for i in range(9):
    m = axs[i, 0].contourf(lon, np.arange(-12, 0, 1),
                           RightNineTerms_Per_Latmean[i, :],
                           cmap=cmap, cmap_kw={'cut': -0.05},
                           levels=np.arange(-0.4, 0.41, 0.05),
                           extend='both')
    
    m = axs[i, 1].contourf(lon, np.arange(-12, 0, 1),
                           RightNineTerms_Trans_Latmean[i, :],
                           cmap=cmap, cmap_kw={'cut': -0.05},
                           levels=np.arange(-0.4, 0.41, 0.05),
                           extend='both')
    
fig.colorbar(m, loc='r', width=0.12, length=0.8, 
             ticklabelsize=15, ticklabelweight='bold',
             extendsize='4em', tickdir='in')

fig.save(r'C:\Users\59799\Desktop\test1.png',
         dpi=600)
plot.close()  # figure的GUI界面经常卡死(图片像素太高),所以直接保存后直接关了
