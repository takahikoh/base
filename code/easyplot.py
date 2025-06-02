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
    """簡易プロットユーティリティ。

    与えられた :class:`pandas.DataFrame` からヒストグラムや散布図などの代表的な
    可視化をウィジェット操作で表示します。Jupyter Notebook 上で以下のように利用
    できます。

    >>> from easyplot import easyplot
    >>> ep = easyplot(df)
    >>> ep.display_all()  # グラフUIが表示される

    DataFrame の各列は数値または ``object`` 型である必要があります。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colnames_list = self.df.columns.to_list()
        self.num_list = self.df.select_dtypes(include=np.number).columns.to_list()
        # 数値列以外をカテゴリ列として保持
        self.cat_list = [col for col in self.colnames_list if col not in self.num_list]

    def _create_filter_widgets(self):
        """列抽出用のウィジェットを生成する。"""
        filter_select = widgets.Dropdown(
            options=['None'] + self.cat_list,
            value='None',
            description='列抽出:'
        )
        filter_values = widgets.SelectMultiple(
            options=[],
            value=(),
            rows=6,
            description='値抽出:'
        )

        def update_values(change):
            if filter_select.value != 'None':
                filter_values.options = natsorted(self.df[filter_select.value].unique())
            else:
                filter_values.options = []

        filter_select.observe(update_values, names='value')
        return filter_select, filter_values

    def _filter_df(self, col: str, vals: tuple):
        """フィルタリングされた DataFrame を返す。"""
        if col != 'None' and vals:
            return self.df[self.df[col].isin(vals)]
        return self.df


    def display_histogram(self):
        x_hist = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="x軸:"
        )
        bins_hist = widgets.BoundedIntText(
            value=30, min=1, max=100, step=1, description="ビン数:"
        )
        filter_select, filter_values = self._create_filter_widgets()

        def disp_hist(x, bins, filter_col, filter_vals):
            if x is None:
                print("表示可能な数値列がありません。")
                return
            data = self._filter_df(filter_col, filter_vals)
            plt.figure(figsize=(10, 5))
            sns.histplot(data[x], bins=bins)
            plt.title(f"{x} のヒストグラム")
            plt.show()

        out_hist = widgets.interactive_output(
            disp_hist,
            {
                'x': x_hist,
                'bins': bins_hist,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([x_hist, bins_hist])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_hist,
        ])
        return ui


    def display_countplot(self):
        x_count = widgets.Dropdown(
            options=self.cat_list,
            value=self.cat_list[0] if self.cat_list else None,
            description="x軸:"
        )
        hue_count = widgets.Dropdown(
            options=['None'] + self.cat_list,
            value='None',
            description="色分け:"
        )
        filter_select, filter_values = self._create_filter_widgets()
        def disp_count(x, hue, filter_col, filter_vals):
            if x is None:
                print("表示可能なカテゴリ列がありません。")
                return
            plt.figure(figsize=(10, 5))
            hue_val = None if hue=='None' else hue
            order = natsorted(self.df[x].dropna().unique())
            data = self._filter_df(filter_col, filter_vals)
            sns.countplot(x=x, hue=hue_val, data=data, order=order)
            plt.title(f"{x} の件数")
            plt.show()
        out_count = widgets.interactive_output(
            disp_count,
            {
                'x': x_count,
                'hue': hue_count,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([x_count, hue_count])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_count,
        ])
        return ui


    def display_barplot(self):
        x_bar = widgets.Dropdown(
            options=self.cat_list,
            value=self.cat_list[0] if self.cat_list else None,
            description="x軸:"
        )
        y_bar = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="y軸:"
        )
        hue_bar = widgets.Dropdown(
            options=['None'] + self.cat_list,
            value='None',
            description="色分け:"
        )
        filter_select, filter_values = self._create_filter_widgets()
        def disp_bar(x, y, hue, filter_col, filter_vals):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            hue_val = None if hue=='None' else hue
            plt.figure(figsize=(10, 5))
            order = natsorted(self.df[x].dropna().unique())
            data = self._filter_df(filter_col, filter_vals)
            sns.barplot(x=x, y=y, hue=hue_val, data=data, order=order)
            plt.title(f"{x} 列における {y} の平均値")
            plt.show()
        out_bar = widgets.interactive_output(
            disp_bar,
            {
                'x': x_bar,
                'y': y_bar,
                'hue': hue_bar,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([x_bar, y_bar, hue_bar])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_bar,
        ])
        return ui
    

    def display_boxplot(self):
        x_box = widgets.Dropdown(
            options=self.cat_list,
            value=self.cat_list[0] if self.cat_list else None,
            description="x軸:"
        )
        y_box = widgets.Dropdown(
            options=self.num_list,
            value=self.num_list[0] if self.num_list else None,
            description="y軸:"
        )
        filter_select, filter_values = self._create_filter_widgets()
        def disp_box(x, y, filter_col, filter_vals):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            plt.figure(figsize=(10, 5))
            order = natsorted(self.df[x].dropna().unique())
            data = self._filter_df(filter_col, filter_vals)
            sns.boxplot(x=x, y=y, data=data, order=order)
            plt.title(f"{x} 列における {y} の箱ひげ図")
            plt.show()
        out_box = widgets.interactive_output(
            disp_box,
            {
                'x': x_box,
                'y': y_box,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([x_box, y_box])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_box,
        ])
        return ui


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
            options=['None'] + self.cat_list,
            value='None',
            description="色分け:"
        )
        filter_select, filter_values = self._create_filter_widgets()
        def disp_scatter(x, y, hue, filter_col, filter_vals):
            if x is None or y is None:
                print("適切な列が選択されていません。")
                return
            hue_val = None if hue=='None' else hue
            plt.figure(figsize=(10, 8))
            data = self._filter_df(filter_col, filter_vals)
            sns.scatterplot(x=x, y=y, hue=hue_val, data=data)
            plt.title(f"{x} と {y} の散布図")
            plt.show()
        out_scatter = widgets.interactive_output(
            disp_scatter,
            {
                'x': x_scatter,
                'y': y_scatter,
                'hue': hue_scatter,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([x_scatter, y_scatter, hue_scatter])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_scatter,
        ])
        return ui

    def display_pairplot(self):
        pair_cols = widgets.SelectMultiple(
            options=self.num_list,
            value=tuple(self.num_list[:min(3, len(self.num_list))]),
            rows=7,
            description="項目:"
        )
        hue_pair = widgets.Dropdown(
            options=['None'] + self.cat_list,
            value='None',
            description="色分け:"
        )
        filter_select, filter_values = self._create_filter_widgets()

        def disp_pair(cols, hue, filter_col, filter_vals):
            if not cols:
                print("項目を選択してください。")
                return
            data = self._filter_df(filter_col, filter_vals)
            hue_val = None if hue == 'None' else hue
            sns.pairplot(data, vars=list(cols), hue=hue_val)
            plt.show()

        out_pair = widgets.interactive_output(
            disp_pair,
            {
                'cols': pair_cols,
                'hue': hue_pair,
                'filter_col': filter_select,
                'filter_vals': filter_values,
            }
        )
        controls = widgets.HBox([pair_cols, hue_pair])
        ui = widgets.VBox([
            controls,
            widgets.HBox([filter_select, filter_values]),
            out_pair,
        ])
        return ui


    def display_correlation_heatmap(self):
        # ウィジェット作成
        corr_cols = widgets.SelectMultiple(
            options=self.colnames_list,
            value=tuple(self.colnames_list[:min(3, len(self.colnames_list))]),
            rows=7,
            description="項目:"
        )
        filter_select = widgets.Dropdown(
            options=['None'] + self.cat_list,
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
        ヒストグラム、Countplot、Barplot、箱ひげ図、散布図、Pairplot をタブ表示し、
        さらに相関ヒートマップを表示します。
        """
        # 各種グラフセクションのヘッダー
        header_graph = widgets.HTML(
            value="""
            <h2 style="text-align:left; margin-bottom: 0;">各種グラフ</h2>
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
        pair_widget = self.display_pairplot()
        corr_ui = self.display_correlation_heatmap()
    
        # タブウィジェットの作成
        tab = widgets.Tab(children=[
            hist_widget, count_widget, bar_widget, box_widget,
            scatter_widget, pair_widget, corr_ui
        ])
        tab_titles = [
            "ヒストグラム", "Countplot", "Barplot",
            "箱ひげ図", "散布図", "Pairplot", "ヒートマップ"
        ]
        for idx, title in enumerate(tab_titles):
            tab.set_title(idx, title)
        display(tab)
        return tab
    
  
