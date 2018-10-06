from ML_actual_combat import get_train_and_test, read_data
from matplotlib import pyplot as plt

get_t = get_train_and_test
housing = read_data.housing


def view_show_message():
    train_housing = get_t.strat_train_set.copy()

    # 加上参数alpha=0.1可以看到数据高密度的区域，
    # 若alpha越靠近0则只加深越高密度的地方。
    # train_housing.plot(kind="scatter", x="longitude",
    #                    y="latitude", alpha=0.1)
    # plt.show()

    """
    为了查看目标median_house_value的在各个地区的分布情况，
    所以可以增加参数c（用于对某参数显示颜色），
    参数s（表示某参数圆的半径），
    cmap表示一个colormap，jet为一个蓝到红的颜色，
    colorbar为右侧颜色条
    """

    train_housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                       s=housing["population"] / 100, label="population",
                       c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()


# 查看特征之间的相关性
def correlation_bet_feature():
    train_housing = get_t.strat_train_set.copy()
    corr_matrix = train_housing.corr()
    result = corr_matrix["median_house_value"].sort_values(ascending=False)
    print(result)


# view_show_message()

correlation_bet_feature()
