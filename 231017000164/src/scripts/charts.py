'''
Draw the last 30 days prediction and real price chart
'''
import pyecharts.options as opts
from pyecharts.charts import Line
import pandas as pd
from datetime import datetime


def get_all_data():
    raw_data = pd.read_csv('data/result.csv')
    print("real_price size", raw_data['real_price'].size)

    return raw_data['date'], raw_data['prediction'], list(raw_data['real_price'])


def generate_all_chart():
    date_list, prediction_list, real_list = get_all_data()
    (
        Line()
        .add_xaxis(xaxis_data=date_list)
        .add_yaxis(
            series_name="Prediction",
            y_axis=prediction_list,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="Real",
            y_axis=real_list,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Bitcoin Prediction"),
            datazoom_opts=opts.DataZoomOpts(type_="slider", range_start=95, range_end=100),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
        .render("templates/bitcoin_all_predict.html")
    )


def main():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
    generate_all_chart()

    print(f'{now}: Chart Done')


if __name__ == '__main__':
    main()
