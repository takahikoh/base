# easyplot.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from natsort import natsorted


class easyplot:
    """
    easyplotは、与えられたpandasのDataFrameに対して
    各種グラフ（ヒストグラム、Countplot、Barplot、箱ひげ図、散布図、相関ヒートマップ）を
    インタラクティブに表示するためのクラスです。
    基本的に各列は数値またはobject型である必要があります。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colnames_list = self.df.columns.to_list()
        self.num_list = self.df.select_dtypes(include=np.number).columns.to_list()
        self.unique_list = [col for col in self.colnames_list if col not in self.num_list]


    def display_histogram(self):
        x_hist = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="x軸:"
        )
        bins_hist = widgets.BoundedIntText(
            value=30, min=1, max=100, step=1, description="ビン数:"
        )
        def disp_hist(x, bins):
            if x is None:
                print("表示可能な数値列がありません。")
                return
            plt.figure(figsize=(10, 5))
            sns.histplot(self.df[x], bins=bins)
            plt.title(f"{x} のヒストグラム")
            plt.show()
        out_hist = widgets.interactive(disp_hist, x=x_hist, bins=bins_hist)
        return out_hist


    def display_countplot(self):
        x_count = widgets.Dropdown(
            options=self.unique_list,
            value=self.unique_list[0] if self.unique_list else None,
            description="x軸:"
        )
        hue_count = widgets.Dropdown(
            options=['None'] + self.unique_list,
            value='None',
            description="色分け:"
        )
        def disp_count(x, hue):
            if x is None:
                print("表示可能なカテゴリ列がありません。")
                return
            plt.figure(figsize=(10, 5))
            hue_val = None if hue=='None' else hue
            order = natsorted(self.df[x].dropna().unique())
            sns.countplot(x=x, hue=hue_val, data=self.df, order=order)
            plt.title(f"{x} の件数")
            plt.show()
        out_count = widgets.interactive(disp_count, x=x_count, hue=hue_count)
        return out_count


    def display_barplot(self):
        x_bar = widgets.Dropdown(
            options=self.unique_list,
            value=self.unique_list[0] if self.unique_list else None,
            description="x軸:"
        )
        y_bar = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="y軸:"
        )
        hue_bar = widgets.Dropdown(
            options=['None'] + self.unique_list,
            value='None',
            description="色分け:"
        )
        def disp_bar(x, y, hue):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            hue_val = None if hue=='None' else hue
            plt.figure(figsize=(10, 5))
            order = natsorted(self.df[x].dropna().unique())
            sns.barplot(x=x, y=y, hue=hue_val, data=self.df, order=order)
            plt.title(f"{x} 列における {y} の平均値")
            plt.show()
        out_bar = widgets.interactive(disp_bar, x=x_bar, y=y_bar, hue=hue_bar)
        return out_bar
    

    def display_boxplot(self):
        x_box = widgets.Dropdown(
            options=self.unique_list,
            value=self.unique_list[0] if self.unique_list else None,
            description="x軸:"
        )
        y_box = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="y軸:"
        )
        def disp_box(x, y):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            plt.figure(figsize=(10, 5))
            order = natsorted(self.df[x].dropna().unique())
            sns.boxplot(x=x, y=y, data=self.df, order=order)
            plt.title(f"{x} 列における {y} の箱ひげ図")
            plt.show()
        out_box = widgets.interactive(disp_box, x=x_box, y=y_box)
        return out_box


    def display_scatterplot(self):
        x_scatter = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="x:"
        )
        y_scatter = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[1] if len(self.num_list) > 1 else None,
            description="y:"
        )
        hue_scatter = widgets.Dropdown(
            options=['None'] + self.unique_list,
            value='None',
            description="色分け:"
        )
        def disp_scatter(x, y, hue):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            hue_val = None if hue=='None' else hue
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=x, y=y, hue=hue_val, data=self.df)
            plt.title(f"{x} と {y} の散布図")
            plt.show()
        out_scatter = widgets.interactive(disp_scatter, x=x_scatter, y=y_scatter, hue=hue_scatter)
        return out_scatter


    def display_correlation_heatmap(self):
        # ウィジェット作成
        corr_cols = widgets.SelectMultiple(
            options=self.colnames_list,
            value=tuple(self.colnames_list[:min(3, len(self.colnames_list))]),
            rows=7,
            description="項目:"
        )
        filter_select = widgets.Dropdown(
            options=['None'] + self.unique_list,
            value='None',
            description='列抽出:'
        )
        filter_values = widgets.SelectMultiple(
            options=[],
            value=(),
            rows=6,
            description='値抽出:'
        )
        def update_filter_values(change):
            if filter_select.value != 'None':
                filter_values.options = natsorted(self.df[filter_select.value].unique())
            else:
                filter_values.options = []
        filter_select.observe(update_filter_values, names='value')
        def disp_corr(filter_col, filter_vals, cols):
            if not cols:
                print("相関を求める項目を選択してください。")
                return
            if filter_col != 'None' and filter_vals:
                data_to_corr = self.df[self.df[filter_col].isin(filter_vals)]
            else:
                data_to_corr = self.df
            cor = data_to_corr[list(cols)].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(cor, cmap=sns.color_palette("coolwarm", 10),
                        annot=True, fmt=".2f", vmin=-1, vmax=1)
            plt.title("相関ヒートマップ")
            plt.show()
        out_corr = widgets.interactive_output(
            disp_corr,
            {'filter_col': filter_select, 'filter_vals': filter_values, 'cols': corr_cols}
        )
        left_box = widgets.VBox([filter_select, filter_values])
        h_box_corr = widgets.HBox([left_box, corr_cols])
        corr_ui = widgets.VBox([h_box_corr, out_corr])
        return corr_ui


    def display_all(self):
        """
        ヒストグラム、Countplot、Barplot、箱ひげ図、散布図をタブ表示し、
        さらに相関ヒートマップを表示します。
        """
        # 各種グラフセクションのヘッダー
        header_graph = widgets.HTML(
            value="""
            <h2 style="text-align:left_box; margin-bottom: 0;">各種グラフ</h2>
            <hr style="border-top: 3px solid #bbb; margin-top: 0; margin-bottom: 20px;">
            """
        )
        display(header_graph)
    
        # 各グラフウィジェットの生成
        hist_widget = self.display_histogram()
        count_widget = self.display_countplot()
        bar_widget = self.display_barplot()
        box_widget = self.display_boxplot()
        scatter_widget = self.display_scatterplot()
        corr_ui = self.display_correlation_heatmap()
    
        # タブウィジェットの作成
        tab = widgets.Tab(children=[
            hist_widget, count_widget, bar_widget, box_widget, scatter_widget, corr_ui
        ])
        tab_titles = ["ヒストグラム", "Countplot", "Barplot", "箱ひげ図", "散布図", "ヒートマップ"]
        for idx, title in enumerate(tab_titles):
            tab.set_title(idx, title)
        display(tab)
    
  
