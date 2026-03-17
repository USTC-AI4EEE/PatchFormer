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
    save_figure_dir = 'figures/model_eval/CS2_35/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_300'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0244, 	0.0369, 	0.0200, 	0.0239, 	0.0109, 	0.0216, 	0.0197, 	0.0141, 	0.0431, 	0.0058 ],  
        'RMSE': [0.0400, 	0.0482, 	0.0320, 	0.0364, 	0.0195, 	0.0336, 	0.0301, 	0.0229, 	0.0640, 	0.0131 ]   
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
    scatter.set_xlim([0.004,0.085])
    scatter.set_ylim([0.008,0.090])
    plt.xticks([0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085],['0.005','','0.025','','0.045','','0.065','','0.085'],fontproperties=font_prop_1a)
    plt.yticks([0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090],['0.010','','0.030','','0.050','','0.070','','0.090'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:
            scatter.text(df['MAE'][line]-0.0135, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.075,0.015,'SP = 300',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
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
    save_figure_dir = 'figures/model_eval/CS2_35/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_400'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0287, 	0.0456, 	0.0230, 	0.0281, 	0.0124, 	0.0242, 	0.0224, 	0.0166, 	0.0532, 	0.0064 ],  
        'RMSE': [0.0441, 	0.0547, 	0.0357, 	0.0402, 	0.0215, 	0.0366, 	0.0328, 	0.0254, 	0.0713, 	0.0146 ]   
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
    scatter.set_xlim([0.004,0.085])
    scatter.set_ylim([0.008,0.090])
    plt.xticks([0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085],['0.005','','0.025','','0.045','','0.065','','0.085'],fontproperties=font_prop_1a)
    plt.yticks([0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090],['0.010','','0.030','','0.050','','0.070','','0.090'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:
            scatter.text(df['MAE'][line]-0.0135, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.075,0.015,'SP = 400',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
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
    save_figure_dir = 'figures/model_eval/CS2_35/scatter_diagram/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'scatter_diagram_SP_500'
    # 创建一个示例 DataFrame
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'MAE': [0.0341, 	0.0559, 	0.0276, 	0.0333, 	0.0147, 	0.0281, 	0.0260, 	0.0198, 	0.0659, 	0.0070 ],  
        'RMSE': [0.0513, 	0.0636, 	0.0422, 	0.0460, 	0.0244, 	0.0410, 	0.0366, 	0.0285, 	0.0825, 	0.0158 ]   
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
    scatter.set_xlim([0.004,0.085])
    scatter.set_ylim([0.008,0.090])
    plt.xticks([0.005,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085],['0.005','','0.025','','0.045','','0.065','','0.085'],fontproperties=font_prop_1a)
    plt.yticks([0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090],['0.010','','0.030','','0.050','','0.070','','0.090'],fontproperties=font_prop_1a)
    # 添加模型名称标签
    for line in range(0, df.shape[0]):
        if line == 2:
            scatter.text(df['MAE'][line]-0.0135, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
        else:
            scatter.text(df['MAE'][line]+0.001, df['RMSE'][line]-0.001, df['Model'][line], horizontalalignment='left', color=sns.color_palette("tab10")[colors_id[line]],fontproperties=font_prop_1c)
    
    scatter.text(0.075,0.015,'SP = 500',horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_1b)
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
    save_figure_dir = 'figures/model_eval/CS2_35/radar_chart/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'r2_radar_chart'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [0.9608, 	0.9469, 	0.9750, 	0.9682, 	0.9909, 	0.9738, 	0.9782, 	0.9880, 	0.9060, 	0.9961 ],
        'SP = 400': [0.9512, 	0.9304, 	0.9683, 	0.9607, 	0.9888, 	0.9681, 	0.9735, 	0.9849, 	0.8814, 	0.9951 ],
        'SP = 500': [0.9245, 	0.8938, 	0.9497, 	0.9416, 	0.9838, 	0.9550, 	0.9628, 	0.9787, 	0.8212, 	0.9935 ]
    }

    df = pd.DataFrame(data)

    # 准备数据
    categories = df['Model'].tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12,9), subplot_kw=dict(polar=True))
    color_dict = {
        'SP = 300': sns.color_palette("tab10")[0],
        'SP = 400': sns.color_palette("tab10")[1],
        'SP = 500': sns.color_palette("tab10")[2],
    }
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for j in np.arange(0.8, 1.0+0.04, 0.04):
        ax.plot(angles, 11*[j], '-', lw=1, color='black')
    for j in range(10):
        ax.plot([angles[j], angles[j]], [0.8, 1.0], '-', lw=1, color='black')
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
    ax.set_rlim(0.8,1.0)
    ax.set_rgrids([0.80,0.84,0.88,0.92,0.96,1.00],fontproperties=font_prop_1a)
    # 设置轴标签
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories,fontproperties=font_prop_1d)
    ax.tick_params(axis='x', pad=20, rotation=0)

    # 添加图例
    ax.legend(loc='upper right',bbox_to_anchor=(1.33,1.10),frameon=True, prop=font_prop_1b,ncol=1)    
    fig.text(0.85, 0.1, 'R',horizontalalignment='center', fontsize=32,fontproperties=font_prop_1c)#fontproperties=font_prop_1c
    fig.text(0.868, 0.12, '2',horizontalalignment='center', fontsize=20,fontproperties=font_prop_1c)
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
    save_figure_dir = 'figures/model_eval/CS2_35/bar_plot/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'AE_bar_plot'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [15.8, 	42.0, 	9.5, 	15.8, 	3.2, 	10.3, 	14.4, 	6.9, 	40.3, 	1.0 ],
        'SP = 400': [16.4, 	44.3, 	11.4, 	18.6, 	3.3, 	11.0, 	15.9, 	8.1, 	41.9, 	0.9 ],
        'SP = 500': [16.7, 	47.0, 	11.5, 	19.9, 	5.3, 	12.0, 	17.6, 	10.9, 	43.4, 	0.9 ]
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
    bar.set_ylim([0.0,140.0])
    plt.xticks(fontproperties=font_prop_1c)
    plt.yticks([0.0,15.0,30.0,45.0,60.0,75.0,90.0,105.0,120.0,135.0],['0.0','15.0','30.0','45.0','60.0','75.0','90.0','105.0','120.0','135.0'],fontproperties=font_prop_1a)
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
    save_figure_dir = 'figures/model_eval/CS2_35/bar_plot/'
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
    save_filename = 'RE_bar_plot'
    # 创建示例数据
    data = {
        'Model': ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours'],
        'SP = 300': [0.0465, 	0.1235, 	0.0279, 	0.0465, 	0.0094, 	0.0303, 	0.0424, 	0.0203, 	0.1185, 	0.0029 ],
        'SP = 400': [0.0683, 	0.1846, 	0.0475, 	0.0775, 	0.0138, 	0.0458, 	0.0662, 	0.0337, 	0.1746, 	0.0037 ],
        'SP = 500': [0.1193, 	0.3357, 	0.0821, 	0.1421, 	0.0379, 	0.0857, 	0.1257, 	0.0779, 	0.3100, 	0.0064 ]
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
    bar.set_xlim([0.00,0.70])
    plt.yticks(fontproperties=font_prop_1c)
    plt.xticks([0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70],['0.00','0.10','0.20','0.30','0.40','0.50','0.60','0.70'],fontproperties=font_prop_1a)
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

