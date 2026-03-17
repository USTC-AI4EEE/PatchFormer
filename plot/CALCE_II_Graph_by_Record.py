import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import os
plt.rcParams['font.family'] = 'Times New Roman'
def draw_all_models_rul_prediction_curve_test_on_CS2_37():
    colors_id = [5,8,6,4,1,0,2,9,7,3]
    palette = sns.color_palette("tab10")
    sorted_palette = [palette[i] for i in colors_id]
    marker_dict = {'real_data':'<',#左箭头
                    'TimesNet':'p',#五边形
                    'TimeMixer':'h',#六边形(长斜边竖直)
                    'PatchTST':'o',#大圆点
                    'MambaSimple':'s',#正方形
                    'ModernTCN':'^',#上箭头
                    'iTransformer':'v',#下箭头
                    'Autoformer':'D',#倒正方形
                    'FEDformer':'d',#小菱形
                    'PathFormer':'X',#小叉
                    'Ours':'>' #右箭头
                    }
    model_name_list = ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours']
    count = 10
    Rated_Capacity = 1.1
    SP_list = ['SP300','SP400','SP500']

    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'   
    font_prop_2 = FontProperties(fname=otf_font_path_2,size=12)  
    save_figure_root = 'figures/curve_graph/test_on_CS2_37/'
    if not os.path.exists(save_figure_root):
        os.makedirs(save_figure_root)
    real_data = torch.load('results/Capacity_{}_real_data.pth'.format('CS2_37'))
    results_RUL_CS2_37_model_dict = {}
    for model in model_name_list:
        results_RUL_CS2_37_model_dict[model] = torch.load('results/RUL_{}_{}.pth'.format('CS2_37',model))
    for SP in SP_list:
        save_figure_dir = os.path.join(save_figure_root,SP)
        if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)
        for i in range(count):
            save_filename = 'Exp_{}'.format(i)+'_RUL_Prediction'
            fig, ax = plt.subplots(1,figsize=(8,6))
            data_0={'cycle':[t+1 for t in range(len(real_data))],'Real data':real_data}
            df_0 = pd.DataFrame(data_0)
            plt.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--',label='failure threshold')
            sns.lineplot(data=df_0,x='cycle',y='Real data',color=sns.color_palette("Paired")[4],label='real data',ax=ax,
                         marker=marker_dict['real_data'],markersize=1e-6,markevery=(1,2000),
                         )
            plt.xlim([0,1000])
            plt.ylim([0.0,1.4])
            plt.xticks([0,100,200,300,400,500,600,700,800,900,1000],['0','','200','','400','','600','','800','','1000'],fontsize=20)
            plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4],['0.0','0.2','0.4','0.6','0.8','1.0','1.2','1.4'],fontsize=20)
            for index,model in enumerate(model_name_list):
                data={'cycle':[t for t in range(int(SP[-3:]),int(SP[-3:])+len(results_RUL_CS2_37_model_dict[model][SP][i]))],
                    'SP = {}'.format(SP[-3:]):results_RUL_CS2_37_model_dict[model][SP][i]}
                df = pd.DataFrame(data)
                plt.axvline(x=int(SP[-3:]),color=mcolors.CSS4_COLORS['magenta'],linestyle='-')
                sns.lineplot(
                    data=df,x='cycle',y='SP = {}'.format(SP[-3:]),
                    # linewidth=1,
                    color=sorted_palette[index],
                    label=model,
                    ax=ax,
                    marker=marker_dict[model],
                    markersize=1e-6,
                    markevery=(1,2000),
                )                                                         
            axins = ax.inset_axes([0.08, 0.08, 0.5, 0.5],transform=ax.transAxes)
            # 在子图中绘制相同的曲线
            axins.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--')
            sns.lineplot(data=df_0,x='cycle',y='Real data',color=sns.color_palette("Paired")[4],ax=axins,
                         marker=marker_dict['real_data'],markersize=6,markevery=(1, 1),linewidth=1)
            for index,model in enumerate(model_name_list):
                data={'cycle':[t for t in range(int(SP[-3:]),int(SP[-3:])+len(results_RUL_CS2_37_model_dict[model][SP][i]))],
                    'SP = {}'.format(SP[-3:]):results_RUL_CS2_37_model_dict[model][SP][i]}
                df = pd.DataFrame(data)
                sns.lineplot(
                    data=df,x='cycle',y='SP = {}'.format(SP[-3:]),
                    linewidth=1,
                    color=sorted_palette[index],
                    # label=model,
                    ax=axins,
                    marker=marker_dict[model],
                    markersize=6,
                    markevery=(1, 1),
                )     
            # 调整子图的坐标轴以聚焦特定区域
            axins.set_xlim(700, 750)
            axins.set_ylim(0.72, 0.82)
            axins.set_xticks([705,715,725,735,745])
            axins.set_yticks([0.74,0.76,0.78,0.80])
            axins.set_xticklabels(['705','715','725','735','745'], fontsize=16)
            axins.set_yticklabels(['0.74','0.76','0.78','0.80'], fontsize=16)
            axins.set_xlabel('')
            axins.set_ylabel('')
            # 添加连接线，指示主图和子图之间的关系
            ax.indicate_inset_zoom(axins, edgecolor="black")
            ax.set_xlabel('Cycle', fontsize=22)
            ax.set_ylabel('Capacity (Ah)', fontsize=22)
            ax.legend(loc='upper center', frameon=True,markerscale=6e+6,fontsize=22,prop=font_prop_2,ncol=4,handletextpad=0.4,columnspacing=0.4) 
            fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
            plt.show()
            plt.close(fig)    

def single_model_single_exp_rul_prediction_curve_test_on_CS2_37(real_data,pred_data_1,pred_data_2,pred_data_3,
    save_filename,save_figure_dir,model):     
    Rated_Capacity = 1.1
    otf_font_path_1 = 'fonts/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径          
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=18)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=14)    
    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'   
    font_prop_2 = FontProperties(fname=otf_font_path_2,size=20)

    palette = sns.color_palette("tab10")
    id_list = {'TimeMixer':'(a) ','TimesNet':'(b) ','PatchTST':'(c) ','MambaSimple':'(d) ','ModernTCN':'(e) ','Autoformer':'(f) ','FEDformer':'(g) ',
               'iTransformer':'(h) ','PathFormer':'(i) ','Ours':'(j) '}

    data_0={'cycle':[t+1 for t in range(len(real_data))],
          'Real data':real_data}
    df_0 = pd.DataFrame(data_0)
    data_1={'cycle':[t for t in range(300,300+len(pred_data_1))],
          'SP = 300':pred_data_1}
    df_1 = pd.DataFrame(data_1)
    data_2={'cycle':[t for t in range(400,400+len(pred_data_2))],
          'SP = 400':pred_data_2}
    df_2 = pd.DataFrame(data_2)
    data_3={'cycle':[t for t in range(500,500+len(pred_data_3))],
          'SP = 500':pred_data_3}
    df_3 = pd.DataFrame(data_3)


    fig,ax = plt.subplots(1,figsize=(8,6))
    sns.lineplot(data=df_0,x='cycle',y='Real data',color=palette[3],label='Real data',linewidth=2,marker='>',markersize=1e-6,markevery=(1,2000),ax=ax)
    sns.lineplot(data=df_1,x='cycle',y='SP = 300',color=palette[0],label="SP = 300",linewidth=2,marker='s',markersize=1e-6,markevery=(1,2000),ax=ax)
    sns.lineplot(data=df_2,x='cycle',y='SP = 400',color=palette[1],label="SP = 400",linewidth=2,marker='o',markersize=1e-6,markevery=(1,2000),ax=ax)
    sns.lineplot(data=df_3,x='cycle',y='SP = 500',color=palette[2],label="SP = 500",linewidth=2,marker='v',markersize=1e-6,markevery=(1,2000),ax=ax)
    plt.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--')
    ax.set_xlim([-50,1050])
    ax.set_ylim([0.1,1.3])
    ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0,1.2])
    ax.set_xticklabels(['0','100','200','300','400','500','600','700','800','900','1000'], fontsize=20)
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0','1.2'], fontsize=20)   
    axins = ax.inset_axes([0.1, 0.1, 0.47, 0.47],transform=ax.transAxes)
    # 在子图中绘制相同的曲线
    axins.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--')
    sns.lineplot(data=df_0,x='cycle',y='Real data',color=palette[3],ax=axins,marker='>',markersize=6,markevery=(1, 1),linewidth=1)
    sns.lineplot(data=df_1,x='cycle',y='SP = 300',color=palette[0],ax=axins,marker='s',markersize=6,markevery=(1, 1),linewidth=1)
    sns.lineplot(data=df_2,x='cycle',y='SP = 400',color=palette[1],ax=axins,marker='o',markersize=6,markevery=(1, 1),linewidth=1)
    sns.lineplot(data=df_3,x='cycle',y='SP = 500',color=palette[2],ax=axins,marker='v',markersize=6,markevery=(1, 1),linewidth=1)
    # 调整子图的坐标轴以聚焦特定区域
    axins.set_xlim(700, 750)
    axins.set_ylim(0.72, 0.82)
    axins.set_xticks([705,715,725,735,745])
    axins.set_yticks([0.74,0.76,0.78,0.80])
    axins.set_xticklabels(['705','715','725','735','745'], fontsize=16)
    axins.set_yticklabels(['0.74','0.76','0.78','0.80'], fontsize=16)
    axins.set_xlabel('')
    axins.set_ylabel('')
    # 添加连接线，指示主图和子图之间的关系
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.text(25,1.2,id_list[model],horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_2)
    ax.text(500,1.2,model,horizontalalignment='center',verticalalignment='center', color='k',fontproperties=font_prop_2)
    ax.set_xlabel('Cycle', fontsize=22)
    ax.set_ylabel('Capacity (Ah)', fontsize=22)
    # lines = [ax.get_lines()[0], ax.get_lines()[1], ax.get_lines()[2], ax.get_lines()[3]]
    ax.legend(loc='upper right',bbox_to_anchor=(1.01,1.01), frameon=True, markerscale=8e+6,prop=font_prop_1a) # lines,['Real data', "SP = 300", "SP = 400", "SP = 500"],
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
    plt.show()
    plt.close(fig)    

def severally_draw_all_models_rul_prediction_curve_test_on_CS2_37():
    model_name_list = ['TimeMixer','TimesNet','PatchTST','MambaSimple','ModernTCN','Autoformer','FEDformer','iTransformer','PathFormer','Ours']
    # model_name_list = ['Ours']
    count = 10
    for model in model_name_list:
        save_figure_dir = os.path.join('figures/curve_graph/single-model_figs/CS2_37/',model)
        if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)
        real_data = torch.load('results/Capacity_{}_real_data.pth'.format('CS2_37'))
        model_predict_data = torch.load('results/RUL_{}_{}.pth'.format('CS2_37',model)) 
        for i in range(count):
            save_filename = 'Exp_{}'.format(i)+'_RUL_Prediction'
            single_model_single_exp_rul_prediction_curve_test_on_CS2_37(real_data,model_predict_data['SP300'][i],
                model_predict_data['SP400'][i],model_predict_data['SP500'][i],save_filename,save_figure_dir,model)

def draw_ablation_models_rul_prediction_curve_CS2_37():
    colors_id = [0,1,2,3]
    palette = sns.color_palette("tab10")
    sorted_palette = [palette[i] for i in colors_id]
    marker_dict = {'real_data':'o',#大圆点
                    'Ours_Patch_Embed':'d',#小菱形
                    'Ours_no_Encoder':'^',#上箭头
                    'Ours_no_SEBlock':'X',#小叉
                    'Ours':'>' #右箭头
                    }
    model_name_list = ['Ours_no_SEBlock','Ours_no_Encoder','Ours_Patch_Embed','Ours']
    label_list = ['w/o SE Block','w/o Encoder','alt. Embedding','Ours']
    count = 10
    Rated_Capacity = 1.1
    SP_list = ['SP300','SP400','SP500']

    otf_font_path_2 = 'fonts/Times-Newer-Roman-Bold.otf'   
    font_prop_2 = FontProperties(fname=otf_font_path_2,size=12)  
    save_figure_root = 'figures/curve_graph/ablation/CS2_37/'
    if not os.path.exists(save_figure_root):
        os.makedirs(save_figure_root)
    real_data = torch.load('results/Capacity_{}_real_data.pth'.format('CS2_37'))
    results_RUL_CS2_37_model_dict = {}
    for model in model_name_list:
        results_RUL_CS2_37_model_dict[model] = torch.load('results/RUL_{}_{}.pth'.format('CS2_37',model))
    for SP in SP_list:
        save_figure_dir = os.path.join(save_figure_root,SP)
        if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)
        for i in range(count):
            save_filename = 'Exp_{}'.format(i)+'_RUL_Prediction'
            fig, ax = plt.subplots(1,figsize=(8,6))
            data_0={'cycle':[t+1 for t in range(len(real_data))],'Real data':real_data}
            df_0 = pd.DataFrame(data_0)
            sns.lineplot(data=df_0,x='cycle',y='Real data',color=sns.color_palette("tab10")[9],label='real data',ax=ax,
                         marker=marker_dict['real_data'],markersize=1e-6,markevery=(1,2000),
                         )
            plt.xlim([0,1000])
            plt.ylim([0.0,1.4])
            plt.xticks([0,100,200,300,400,500,600,700,800,900,1000],['0','','200','','400','','600','','800','','1000'],fontsize=20)
            plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4],['0.0','0.2','0.4','0.6','0.8','1.0','1.2','1.4'],fontsize=20)
            for index,model in enumerate(model_name_list):
                data={'cycle':[t for t in range(int(SP[-3:]),int(SP[-3:])+len(results_RUL_CS2_37_model_dict[model][SP][i]))],
                    'SP = {}'.format(SP[-3:]):results_RUL_CS2_37_model_dict[model][SP][i]}
                df = pd.DataFrame(data)
                sns.lineplot(
                    data=df,x='cycle',y='SP = {}'.format(SP[-3:]),
                    # linewidth=1,
                    color=sorted_palette[index],
                    label=label_list[index],
                    ax=ax,
                    marker=marker_dict[model],
                    markersize=1e-6,
                    markevery=(1,2000),
                )           
            plt.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--',label='failure threshold')    
            plt.axvline(x=int(SP[-3:]),color=sns.color_palette("tab10")[4],linestyle='-')                                          
            axins = ax.inset_axes([0.08, 0.08, 0.5, 0.5],transform=ax.transAxes)
            # 在子图中绘制相同的曲线
            axins.axhline(y=Rated_Capacity*0.7,color='black',linestyle='--')
            sns.lineplot(data=df_0,x='cycle',y='Real data',color=sns.color_palette("tab10")[9],ax=axins,
                         marker=marker_dict['real_data'],markersize=6,markevery=(1, 1),linewidth=1)
            for index,model in enumerate(model_name_list):
                data={'cycle':[t for t in range(int(SP[-3:]),int(SP[-3:])+len(results_RUL_CS2_37_model_dict[model][SP][i]))],
                    'SP = {}'.format(SP[-3:]):results_RUL_CS2_37_model_dict[model][SP][i]}
                df = pd.DataFrame(data)
                sns.lineplot(
                    data=df,x='cycle',y='SP = {}'.format(SP[-3:]),
                    linewidth=1,
                    color=sorted_palette[index],
                    # label=model,
                    ax=axins,
                    marker=marker_dict[model],
                    markersize=6,
                    markevery=(1, 1),
                )     
            # 调整子图的坐标轴以聚焦特定区域
            axins.set_xlim(700, 750)
            axins.set_ylim(0.72, 0.82)
            axins.set_xticks([705,715,725,735,745])
            axins.set_yticks([0.74,0.76,0.78,0.80])
            axins.set_xticklabels(['705','715','725','735','745'], fontsize=16)
            axins.set_yticklabels(['0.74','0.76','0.78','0.80'], fontsize=16)
            axins.set_xlabel('')
            axins.set_ylabel('')
            # 添加连接线，指示主图和子图之间的关系
            ax.indicate_inset_zoom(axins, edgecolor="black")
            ax.set_xlabel('Cycle', fontsize=22)
            ax.set_ylabel('Capacity (Ah)', fontsize=22)
            ax.legend(loc='upper right', frameon=True,markerscale=6e+6,fontsize=22,prop=font_prop_2,ncol=1,handletextpad=0.4,columnspacing=0.4) 
            fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
            plt.show()
            plt.close(fig)    


if __name__ == '__main__':
    draw_all_models_rul_prediction_curve_test_on_CS2_37()
    severally_draw_all_models_rul_prediction_curve_test_on_CS2_37()
    draw_ablation_models_rul_prediction_curve_CS2_37()
