import subprocess
import os
import sys
import time

# ================= 配置 =================
EXE_PATH = r".\\libs\\llm_demo.exe"  # 请确保路径正确
MODEL_CONFIG = r".\\models\\Qwen3-1.7B-MNN\\config.json"  # 请确保路径正确
MAX_NEW_TOKENS = "64"

# 100个 Prompt 列表
prompts = [
    "设计一个科技公司官网首页，直接输出html代码. </no_think>",
    "设计一个电商平台首页，直接输出html代码. </no_think>",
    "设计一个企业级SaaS后台管理系统仪表盘，直接输出html代码. </no_think>",
    "设计一个新闻资讯类网站首页，直接输出html代码. </no_think>",
    "设计一个CRM客户关系管理系统的客户列表页，直接输出html代码. </no_think>",
    "设计一个ERP企业资源计划系统的库存管理界面，直接输出html代码. </no_think>",
    "设计一个数字货币钱包的资产概览页，直接输出html代码. </no_think>",
    "设计一个NFT交易市场的艺术品展示页，直接输出html代码. </no_think>",
    "设计一个在线协作白板的工具栏界面，直接输出html代码. </no_think>",
    "设计一个代码托管平台的代码审查界面，直接输出html代码. </no_think>",
    "设计一个网络安全监控系统的大屏可视化界面，直接输出html代码. </no_think>",
    "设计一个智慧城市交通管理系统的监控中心界面，直接输出html代码. </no_think>",
    "设计一个餐饮点餐系统的iPad自助终端界面，直接输出html代码. </no_think>",
    "设计一个电影票务预订App的选座界面，直接输出html代码. </no_think>",
    "设计一个在线拍卖平台的实时竞价界面，直接输出html代码. </no_think>",
    "设计一个婚恋交友App的匹配滑动界面，直接输出html代码. </no_think>",
    "设计一个母婴育儿社区的论坛帖子详情页，直接输出html代码. </no_think>",
    "设计一个宠物服务平台的寄养预约界面，直接输出html代码. </no_think>",
    "设计一个法律咨询平台的律师咨询聊天界面，直接输出html代码. </no_think>",
    "设计一个在线翻译工具的双语对照界面，直接输出html代码. </no_think>",
    "设计一个计算器App的高级科学计算模式界面，直接输出html代码. </no_think>",
    "设计一个日历日程管理应用周视图界面，直接输出html代码. </no_think>",
    "设计一个食谱大全App的烹饪步骤详情页，直接输出html代码. </no_think>",
    "设计一个时尚电商的虚拟试衣间界面，直接输出html代码. </no_think>",
    "设计一个美妆品牌的AR试妆功能界面，直接输出html代码. </no_think>",
    "设计一个游戏直播平台的直播间界面，直接输出html代码. </no_think>",
    "设计一个电竞战队官网的成员介绍页，直接输出html代码. </no_think>",
    "设计一个物流快递员专用的扫码揽件界面，直接输出html代码. </no_think>",
    "设计一个在线问卷调查工具的编辑页面，直接输出html代码. </no_think>",
    "设计一个知识付费平台的课程购买落地页，直接输出html代码. </no_think>",
    "设计一个个人博客网站的文章归档页，直接输出html代码. </no_think>",
    "设计一个设计师作品集网站的项目展示页，直接输出html代码. </no_think>",
    "设计一个移动应用的启动引导页（Onboarding），直接输出html代码. </no_think>",
    "设计一个通用的用户登录与注册页面，直接输出html代码. </no_think>",
    "设计一个电商平台的购物车结算流程页面，直接输出html代码. </no_think>",
    "设计一个App的侧边导航栏（Sidebar）菜单，直接输出html代码. </no_think>",
    "设计一个网站的404错误提示页面，直接输出html代码. </no_think>",
    "设计一个系统的全局搜索结果下拉框，直接输出html代码. </no_think>",
    "设计一个应用的个人账号设置页面，直接输出html代码. </no_think>",
    "设计一个消息通知中心的列表界面，直接输出html代码. </no_think>",
    "设计一个带有筛选功能的商品分类列表页，直接输出html代码. </no_think>",
    "设计一个用户反馈与评价的输入表单，直接输出html代码. </no_think>",
    "设计一个AI聊天机器人（Chatbot）的对话窗口，直接输出html代码. </no_think>",
    "设计一个生成式AI工具的Prompt输入与结果展示页，直接输出html代码. </no_think>",
    "设计一个VR虚拟现实设备的主菜单界面，直接输出html代码. </no_think>",
    "设计一个车载中控系统的导航与娱乐分屏界面，直接输出html代码. </no_think>",
    "设计一个智能手表的健康监测主界面，直接输出html代码. </no_think>",
    "设计一个无人机操控App的飞行参数仪表盘，直接输出html代码. </no_think>",
    "设计一个3D打印机的控制软件界面，直接输出html代码. </no_think>",
    "设计一个物联网（IoT）设备的连接配对界面，直接输出html代码. </no_think>",
    "设计一个自助银行柜员机（ATM）的操作界面，直接输出html代码. </no_think>",
    "设计一个超市自助结账机（Kiosk）的触摸屏界面，直接输出html代码. </no_think>",
    "设计一个慈善众筹平台的项目捐款页，直接输出html代码. </no_think>",
    "设计一个图书馆管理系统的图书检索页，直接输出html代码. </no_think>",
    "设计一个博物馆导览App的展品解说页，直接输出html代码. </no_think>",
    "设计一个艺术画廊的线上虚拟展厅入口，直接输出html代码. </no_think>",
    "设计一个心理健康冥想App的播放背景页，直接输出html代码. </no_think>",
    "设计一个睡眠监测App的睡眠质量分析报告页，直接输出html代码. </no_think>",
    "设计一个喝水提醒App的打卡界面，直接输出html代码. </no_think>",
    "设计一个女性生理期记录App的日历界面，直接输出html代码. </no_think>",
    "设计一个育儿疫苗接种提醒的时间轴界面，直接输出html代码. </no_think>",
    "设计一个老年人专用的大字体通讯录界面，直接输出html代码. </no_think>",
    "设计一个企业内部OA办公系统的审批流程页，直接输出html代码. </no_think>",
    "设计一个进销存管理系统的报表分析页，直接输出html代码. </no_think>",
    "设计一个客服工单系统的待处理工单列表，直接输出html代码. </no_think>",
    "设计一个开发者API文档的导航与阅读界面，直接输出html代码. </no_think>",
    "设计一个服务器运维面板的资源监控页，直接输出html代码. </no_think>",
    "设计一个数据可视化BI工具的拖拽编辑画板，直接输出html代码. </no_think>",
    "设计一个视频剪辑软件的时间轴轨道界面，直接输出html代码. </no_think>",
    "设计一个图片修图App的滤镜调节面板，直接输出html代码. </no_think>",
    "设计一个矢量绘图工具的图层管理面板，直接输出html代码. </no_think>",
    "设计一个音乐制作软件（DAW）的混音台界面，直接输出html代码. </no_think>",
    "设计一个星座运势App的每日运程页，直接输出html代码. </no_think>",
    "设计一个彩票购买App的开奖结果页，直接输出html代码. </no_think>",
    "设计一个婚礼策划助手的待办事项清单，直接输出html代码. </no_think>",
    "设计一个装修设计App的3D样板间漫游页，直接输出html代码. </no_think>",
    "设计一个花店小程序的鲜花定制页面，直接输出html代码. </no_think>",
    "设计一个酒庄官网的红酒品鉴介绍页，直接输出html代码. </no_think>",
    "设计一个极限运动俱乐部的活动报名页，直接输出html代码. </no_think>",
    "设计一个寻人/寻物启事平台的信息发布页，直接输出html代码. </no_think>",
    "设计一个环保回收App的垃圾分类指南页，直接输出html代码. </no_think>",
    "设计一个时间胶囊App的未来信件封存界面，直接输出html代码. </no_think>",
    "设计一个多人聚餐AA制分账App的账单结算页，直接输出html代码. </no_think>",
    "设计一个税务申报系统的年度报税进度条，直接输出html代码. </no_think>",
    "设计一个个人信用评分查询App的仪表盘，直接输出html代码. </no_think>",
    "设计一个保险理赔App的事故现场照片上传页，直接输出html代码. </no_think>",
    "设计一个货币汇率转换器的实时计算界面，直接输出html代码. </no_think>",
    "设计一个退休养老金规划计算器的结果展示页，直接输出html代码. </no_think>",
    "设计一个股票交易App的K线图全屏查看模式，直接输出html代码. </no_think>",
    "设计一个贷款申请App的额度预估页面，直接输出html代码. </no_think>",
    "设计一个电子发票管理系统的发票详情页，直接输出html代码. </no_think>",
    "设计一个校园卡充值小程序的支付页面，直接输出html代码. </no_think>",
    "设计一个背单词App的记忆曲线复习界面，直接输出html代码. </no_think>",
    "设计一个专注力（番茄钟）工具的计时倒数界面，直接输出html代码. </no_think>",
    "设计一个思维导图工具的节点编辑菜单，直接输出html代码. </no_think>",
    "设计一个在线考试系统的防作弊摄像头监控提示页，直接输出html代码. </no_think>",
    "设计一个时间胶囊App的未来信件封存界面，直接输出html代码. </no_think>",
    "设计一个多人聚餐AA制分账App的账单结算页，直接输出html代码. </no_think>",
    "设计一个税务申报系统的年度报税进度条，直接输出html代码. </no_think>",
    "设计一个个人信用评分查询App的仪表盘，直接输出html代码. </no_think>"
]

# 存储所有正在运行的进程句柄
running_processes = []

print(f"准备同时发射 {len(prompts)} 个进程...")
time.sleep(1)

start_time = time.time()

for i, prompt in enumerate(prompts):
    output_file = f"result_{i}.json"

    # 构造命令
    cmd = [EXE_PATH, MODEL_CONFIG, prompt, MAX_NEW_TOKENS, output_file]

    try:
        # Popen 是非阻塞的，它启动进程后立即返回，不会等待程序结束
        # 这样就能实现“同时”启动
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # 丢弃输出以减少IO干扰
            stderr=subprocess.DEVNULL)
        running_processes.append(p)
        print(f"已发射: {i}")
    except Exception as e:
        print(f"启动失败 {i}: {e}")

print(f"\n所有 {len(prompts)} 个进程已发出启动指令！")
print("系统正在承受极限负载，请耐心等待...")

# 等待所有进程结束
for i, p in enumerate(running_processes):
    p.wait()

end_time = time.time()
print(f"所有任务结束。总耗时: {end_time - start_time:.2f} 秒")
