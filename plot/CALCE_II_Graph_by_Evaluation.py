import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
plt.rcParams['font.family'] = 'Times New Roman'
def draw_mae_rmse_scatter_diagram_SP_300():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=20)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=22)    
    font_prop_1c = FontProperties(fname=otf_font_path_2,size=16)
    save_figure_dir = 'figures/model_eval/CS2_37/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_300'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0145, 	0.0267, 	0.0121, 	0.0151, 	0.0067, 	0.0168, 	0.0129, 	0.0076, 	0.0256, 	0.0048 ],  
        'RMSE': [0.0188, 	0.0326, 	0.0163, 	0.0197, 	0.0100, 	0.0245, 	0.0175, 	0.0109, 	0.0314, 	0.0086 ]   
    }
    colors_id = [5,8,6,4,1,0,2,9,7,3]
    palette = sns.color_palette("tab10")
    sorted_palette = [palette[i] for i in colors_id]
    df = pd.DataFrame(data)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建散点图
    fig = plt.figure(figsize=(8,6))
    scatter = sns.scatterplot(data=df, x='MAE', y='RMSE',hue='Model',palette=sorted_palette,markers=True, s=50,legend=False)#hue='Model',style='Model',
    scatter.grid(linestyle='--')
    # 设置散点图边框颜色为黑色
    scatter.axes.spines['top'].set_color('black')
    scatter.axes.spines['bottom'].set_color('black')
    scatter.axes.spines['left'].set_color('black')
    scatter.axes.spines['right'].set_color('black')
    scatter.set_xlim([-0.001,0.050])
    scatter.set_ylim([-0.001,0.050])
    plt.xticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    plt.yticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:# PatchTST
            scatter.text(df['MAE'][line]-0.0085, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 3:# MambaSimple
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 0:# TimeMixer
            scatter.text(df['MAE'][line]-0.0097, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 8:# PathFormer
            scatter.text(df['MAE'][line]-0.0105, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 7:# iTransformer
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line], df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.0425,0.005,'SP = 300',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
    # 添加图表标题和轴标签
    # plt.title('Model Performance Comparison')
    plt.xlabel('MAE',fontproperties=font_prop_1b)
    plt.ylabel('RMSE',fontproperties=font_prop_1b)
    # plt.legend(frameon=False)

    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)    

def draw_mae_rmse_scatter_diagram_SP_400():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=20)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=22)    
    font_prop_1c = FontProperties(fname=otf_font_path_2,size=16)
    save_figure_dir = 'figures/model_eval/CS2_37/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_400'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0159, 	0.0312, 	0.0132, 	0.0170, 	0.0073, 	0.0184, 	0.0143, 	0.0082, 	0.0300, 	0.0052 ],  
        'RMSE': [0.0204, 	0.0378, 	0.0175, 	0.0216, 	0.0107, 	0.0264, 	0.0192, 	0.0116, 	0.0364, 	0.0092 ]   
    }
    colors_id = [5,8,6,4,1,0,2,9,7,3]
    palette = sns.color_palette("tab10")
    sorted_palette = [palette[i] for i in colors_id]
    df = pd.DataFrame(data)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建散点图
    fig = plt.figure(figsize=(8,6))
    scatter = sns.scatterplot(data=df, x='MAE', y='RMSE',hue='Model',palette=sorted_palette,markers=True, s=50,legend=False)
    scatter.grid(linestyle='--')
    # 设置散点图边框颜色为黑色
    scatter.axes.spines['top'].set_color('black')
    scatter.axes.spines['bottom'].set_color('black')
    scatter.axes.spines['left'].set_color('black')
    scatter.axes.spines['right'].set_color('black')
    scatter.set_xlim([-0.001,0.050])
    scatter.set_ylim([-0.001,0.050])
    plt.xticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    plt.yticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:# PatchTST
            scatter.text(df['MAE'][line]-0.0085, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 3:# MambaSimple
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 0:# TimeMixer
            scatter.text(df['MAE'][line]-0.0097, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 8:# PathFormer
            scatter.text(df['MAE'][line]-0.0105, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 7:# iTransformer
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line], df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.0425,0.005,'SP = 400',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
    # 添加图表标题和轴标签
    # plt.title('Model Performance Comparison')
    plt.xlabel('MAE',fontproperties=font_prop_1b)
    plt.ylabel('RMSE',fontproperties=font_prop_1b)
    # plt.legend(frameon=False)

    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig) 

def draw_mae_rmse_scatter_diagram_SP_500():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=20)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=22)    
    font_prop_1c = FontProperties(fname=otf_font_path_2,size=16)
    save_figure_dir = 'figures/model_eval/CS2_37/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_500'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0186, 	0.0395, 	0.0152, 	0.0202, 	0.0083, 	0.0211, 	0.0170, 	0.0092, 	0.0379, 	0.0058 ],  
        'RMSE': [0.0230, 	0.0455, 	0.0194, 	0.0246, 	0.0120, 	0.0292, 	0.0218, 	0.0129, 	0.0440, 	0.0098 ]   
    }
    colors_id = [5,8,6,4,1,0,2,9,7,3]
    palette = sns.color_palette("tab10")
    sorted_palette = [palette[i] for i in colors_id]
    df = pd.DataFrame(data)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建散点图
    fig = plt.figure(figsize=(8,6))
    scatter = sns.scatterplot(data=df, x='MAE', y='RMSE',hue='Model',palette=sorted_palette,markers=True, s=50,legend=False)#hue='Model',style='Model',
    scatter.grid(linestyle='--')
    # 设置散点图边框颜色为黑色
    scatter.axes.spines['top'].set_color('black')
    scatter.axes.spines['bottom'].set_color('black')
    scatter.axes.spines['left'].set_color('black')
    scatter.axes.spines['right'].set_color('black')
    scatter.set_xlim([-0.001,0.050])
    scatter.set_ylim([-0.001,0.050])
    plt.xticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    plt.yticks([0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050],['0.000','','0.010','','0.020','','0.030','','0.040','','0.050'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:# PatchTST
            scatter.text(df['MAE'][line]-0.0085, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 3:# MambaSimple
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 0:# TimeMixer
            scatter.text(df['MAE'][line]-0.0097, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 8:# PathFormer
            scatter.text(df['MAE'][line]-0.0105, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        elif line == 7:# iTransformer
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line], df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.0007, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.0425,0.005,'SP = 500',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
    # 添加图表标题和轴标签
    # plt.title('Model Performance Comparison')
    plt.xlabel('MAE',fontproperties=font_prop_1b)
    plt.ylabel('RMSE',fontproperties=font_prop_1b)
    # plt.legend(frameon=False)

    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)  

def draw_r2_radar_chart():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_2,size=20)  # 16    
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=25)  # 20
    font_prop_1c = FontProperties(fname=otf_font_path_1,size=32)  # 25
    font_prop_1d = FontProperties(fname=otf_font_path_2,size=25)
    save_figure_dir = 'figures/model_eval/CS2_37/radar_chart/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'r2_radar_chart'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [0.9913, 	0.9749, 	0.9931, 	0.9902, 	0.9976, 	0.9846, 	0.9923, 	0.9972, 	0.9765, 	0.9983 ],
        'SP = 400': [0.9895, 	0.9659, 	0.9917, 	0.9879, 	0.9973, 	0.9819, 	0.9906, 	0.9967, 	0.9679, 	0.9980 ],
        'SP = 500': [0.9860, 	0.9481, 	0.9889, 	0.9834, 	0.9964, 	0.9769, 	0.9870, 	0.9958, 	0.9509, 	0.9976 ]
    }
    
    df = pd.DataFrame(data)

    # 准备数据
    categories = df['Model'].tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(polar=True))
    color_dict = {
        'SP = 300': sns.color_palette("tab10")[0],
        'SP = 400': sns.color_palette("tab10")[1],
        'SP = 500': sns.color_palette("tab10")[2],
    }
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for j in np.arange(0.94, 1.0+0.01, 0.01):
        ax.plot(angles, 11*[j], '-', lw=1, color='black')
    for j in range(10):
        ax.plot([angles[j], angles[j]], [0.94, 1.0], '-', lw=1, color='black')
    # 绘制数据
    for col in ['SP = 300','SP = 400','SP = 500']:
        values = df[col].tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, linewidth=3, linestyle='solid', marker='s',label=col,color=color_dict[col])
        ax.fill(angles, values, alpha=0.2)
    # 隐藏最外圈的圆
    ax.spines['polar'].set_visible(False)
    # 隐藏圆形网格线
    ax.grid(False)
    ax.set_rlim(0.94,1.0)
    ax.set_rgrids([0.94,0.95,0.96,0.97,0.98,0.99,1.00],fontproperties=font_prop_1a)
    # 设置轴标签
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories,fontproperties=font_prop_1d)
    ax.tick_params(axis='x', pad=20, rotation=0)

    # 添加图例
    ax.legend(loc='upper right',bbox_to_anchor=(1.33,1.10),frameon=True, prop=font_prop_1b,ncol=1)    #bbox_to_anchor=(1.3,1.18),
    # fig.text(0.9, 0.1, r'$\mathrm{R^2}$',horizontalalignment='center', fontsize=32)#fontproperties=font_prop_1c
    fig.text(0.85, 0.1, 'R',horizontalalignment='center', fontproperties=font_prop_1c)#fontproperties=font_prop_1c
    fig.text(0.868, 0.12, '2',horizontalalignment='center', fontsize=20, fontproperties=font_prop_1c)
    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)   

def draw_AE_bar_plot():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=20)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=22)    
    font_prop_1c = FontProperties(fname=otf_font_path_2,size=16)
    save_figure_dir = 'figures/model_eval/CS2_37/bar_plot/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'AE_bar_plot'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [35.8, 	46.6, 	23.7, 	32.1, 	2.4, 	13.1, 	23.4, 	2.5, 	42.2, 	1.0 ],
        'SP = 400': [30.8, 	49.2, 	23.4, 	32.6, 	2.6, 	13.9, 	25.1, 	3.2, 	43.8, 	1.0 ],
        'SP = 500': [31.2, 	54.2, 	23.6, 	33.3, 	4.4, 	15.6, 	28.6, 	5.2, 	47.7, 	1.2 ]
    }

    df = pd.DataFrame(data)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建堆叠柱状图
    fig=plt.figure(figsize=(8, 6))
    bar=sns.barplot(data=df, x='Model', y='SP = 300', color=sns.color_palette("tab10")[0], label='SP = 300',edgecolor='black')
    bar=sns.barplot(data=df, x='Model', y='SP = 400', color=sns.color_palette("tab10")[1], label='SP = 400', bottom=df['SP = 300'],edgecolor='black')
    bar=sns.barplot(data=df, x='Model', y='SP = 500', color=sns.color_palette("tab10")[2], label='SP = 500', bottom=df['SP = 400']+df['SP = 300'],edgecolor='black')

    bar.axes.spines['top'].set_color('black')
    bar.axes.spines['bottom'].set_color('black')
    bar.axes.spines['left'].set_color('black')
    bar.axes.spines['right'].set_color('black')
    bar.set_ylim([0.0,155.0])
    plt.xticks(fontproperties=font_prop_1c)
    plt.yticks([0.0,15.0,30.0,45.0,60.0,75.0,90.0,105.0,120.0,135.0,150.0],['0.0','','30.0','','60.0','','90.0','','120.0','','150.0'],fontproperties=font_prop_1a)
    # 添加图表标题和轴标签
    # plt.title('Model Performance Comparison')
    # plt.xlabel('Models',fontproperties=font_prop_1b)
    plt.xlabel('')
    plt.ylabel('AE',fontproperties=font_prop_1b)

    # 添加图例
    plt.legend(loc='upper center',frameon=True, prop=font_prop_1a,ncol=1)
    # 使x轴的刻度标签倾斜显示
    plt.xticks(rotation=30,ha='right')  # 可以调整角度以适应您的需求
    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)  

def draw_RE_bar_plot():
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径     
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'     
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=20)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=22)    
    font_prop_1c = FontProperties(fname=otf_font_path_2,size=16)
    save_figure_dir = 'figures/model_eval/CS2_37/bar_plot/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'RE_bar_plot'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [0.0861, 	0.1120, 	0.0570, 	0.0772, 	0.0058, 	0.0315, 	0.0563, 	0.0060, 	0.1014, 	0.0024 ],
        'SP = 400': [0.0975, 	0.1557, 	0.0741, 	0.1032, 	0.0082, 	0.0440, 	0.0794, 	0.0101, 	0.1386, 	0.0032 ],
        'SP = 500': [0.1444, 	0.2509, 	0.1093, 	0.1542, 	0.0204, 	0.0722, 	0.1324, 	0.0241, 	0.2208, 	0.0056 ]
    }

    df = pd.DataFrame(data)

    # 设置绘图样式
    sns.set(style="whitegrid")

    # 创建堆叠柱状图
    fig=plt.figure(figsize=(8, 6))
    bar=sns.barplot(data=df, y='Model', x='SP = 300', color=sns.color_palette("tab10")[0], label='SP = 300',edgecolor='black')
    bar=sns.barplot(data=df, y='Model', x='SP = 400', color=sns.color_palette("tab10")[1], label='SP = 400', left=df['SP = 300'],edgecolor='black')
    bar=sns.barplot(data=df, y='Model', x='SP = 500', color=sns.color_palette("tab10")[2], label='SP = 500', left=df['SP = 400']+df['SP = 300'],edgecolor='black')

    bar.axes.spines['top'].set_color('black')
    bar.axes.spines['bottom'].set_color('black')
    bar.axes.spines['left'].set_color('black')
    bar.axes.spines['right'].set_color('black')
    bar.set_xlim([0.00,0.60])
    plt.yticks(fontproperties=font_prop_1c)
    plt.xticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60],['0.00','','0.10','','0.20','','0.30','','0.40','','0.50','','0.60'],fontproperties=font_prop_1a)
    # 添加图表标题和轴标签
    # plt.title('Model Performance Comparison')
    # plt.xlabel('Models',fontproperties=font_prop_1b)
    plt.ylabel('')
    plt.xlabel('RE',fontproperties=font_prop_1b)

    # 添加图例
    plt.legend(loc='center right',frameon=True, prop=font_prop_1a,ncol=1)
    # 使x轴的刻度标签倾斜显示
    plt.yticks(ha='right')  # 可以调整角度以适应您的需求
    # 显示图表
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)  

if __name__ == '__main__':
    draw_mae_rmse_scatter_diagram_SP_300()
    draw_mae_rmse_scatter_diagram_SP_400()
    draw_mae_rmse_scatter_diagram_SP_500()
    draw_r2_radar_chart()
    draw_AE_bar_plot()
    draw_RE_bar_plot()

