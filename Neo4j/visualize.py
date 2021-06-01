# coding:utf-8
from Neo4j.search import level2treat, input2related
from pyecharts import options as opts
from pyecharts.charts import Graph
import webbrowser
import os


def visualize(input):
    action_node, alarm_node, spot_node, level_node, treat_node = input2related(input)

    categories = [
        {"name": "alarm"},
        {"name": "action"},
        {"name": "spot"},
        {"name": "level"},
        {"name": "treat"}
    ]

    # 长度等于1时;大于1时,直接调用
    def noe2echart(neo, category):
        echart = []
        if len(neo) == 1:
            name = str(neo['name'])
            echart.append({"name": name, "symbolSize": 50, 'category': category})
        else:
            name = []
            for i in neo:
                if i['name'] not in name:
                    name.append(i['name'])
            for i in range(len(name)):
                echart.append({"name": name[i], "symbolSize": 50, 'category': category})
        return echart

    alarm = noe2echart(alarm_node, 0)
    action = noe2echart(action_node, 1)
    spot = noe2echart(spot_node, 2)
    level = noe2echart(level_node, 3)
    treat = noe2echart(treat_node, 4)

    # print(action, alarm, spot, level, treat, sep='\n')

    action_links = []
    alarm_links = []
    level_links = []
    spot_links = []
    treat_links = []

    links = []
    nodes = []
    for i in alarm:
        for j in level:
            links.append({"source": i.get("name"), "target": j.get("name")})
    #for i in alarm:
        for j in action:
            links.append({"source": i.get("name"), "target": j.get("name")})
    #for i in alarm:
        for j in spot:
            links.append({"source": i.get("name"), "target": j.get("name")})
    for i in range(len(spot)):
        links.append({"source": action[0].get("name"), "target": spot[i].get("name")})

    # count_dit = {}
    # for i in treat_node:
    #     count_dit[i] = count_dit.get(i, 0) + 1
    # count_list = []
    # k = 0
    # for values in count_dit.values():
    #     count_list.append(values)
    # k = 0
    # for i in range(len(count_list)):
    #     for j in range(count_list[i]):
    #         links.append({"source": level[k].get("name"), "target": treat[i].get("name")})
    #         k += 1

    def merge_node(collect, block):
        for i in block:
            collect.append(i)

    merge_node(nodes, alarm)
    merge_node(nodes, action)
    # merge_node(nodes, level)
    merge_node(nodes, spot)
    # merge_node(nodes, treat)

    c = (
        Graph(init_opts=opts.InitOpts(width='1800px', height='900px',
                                      animation_opts=opts.AnimationOpts(animation_duration=10, animation=False,
                                                                        animation_duration_update=0)))
            .add("",
                 nodes,
                 links,
                 categories=categories,
                 repulsion=800,
                 is_draggable=True,
                 gravity=0.02
                 # edge_label=opts.LabelOpts(is_show=True,
                 #                           position="middle",
                 #                           formatter=" "
                 #                           )
                 )
            .set_global_opts(title_opts=opts.TitleOpts(title="{}-警情图谱".format(input)))
            .render("./graph_base.html")
    )

    abspath = os.path.dirname(os.path.abspath(__file__))
    path = "{}/graph_base.html".format(abspath)
    return path


if __name__ == '__main__':
    input = '短时间打架'
    webbrowser.open(visualize(input))
