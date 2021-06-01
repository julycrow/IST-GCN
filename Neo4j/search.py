# coding:utf-8
from py2neo import Graph, Node, Relationship, NodeSelector


def level2treat(level_node):  # 每个警情程度对应的处置方式
    graph = Graph('bolt://localhost:7687', username='neo4j', password='0000')
    level_name = level_node['name']
    treat = graph.run("match ({{name:'{0}'}})-[r:采取]-(a) return a".format(level_name))
    treat_node = treat.data()[0]['a']
    return treat_node


def input2related(input):
    """
    :param: input: 输入--str类型  egg: input = '长时间打架'

    :return: action_node: 小车行动--单个节点
    :return: alarm_node: 警情--单个节点
    :return: spot_node: 现场行动--多个节点
    :return: level_node: 警情程度--多个节点
    :return: treat_node: 处置方式--多个节点(与level-node相对应,个数相同)
    """
    graph = Graph('bolt://localhost:7687', username='neo4j', password='0000')
    # input = '长时间打架'
    a = 'a'
    b = 'b'

    action = graph.run(
        "match ({{name:'{0}'}})-[r:采取]-({1}), ({2})-[r2:输入分类]-(输入{{name:'{0}'}}) return {1},{2}".format(input, a, b))

    data = action.data()
    if not data:  #
        raise ValueError('输入不正确')
        # print('输入不正确')
        # return
    action_node = data[0][a]  # Node:小车行动:通知警员/声光警告
    alarm_node = data[0][b]  # Node:警情:打架/...
    alarm_name = alarm_node['name']
    action_name = action_node['name']

    spot = graph.run(
        "match ({{name:'{0}'}})-[r1:执行]-({2}), ({{name:'{1}'}})-[r2:执行]-({2}) return {2}".format(alarm_name,
                                                                                                 action_name, a))
    spot_node = [_[a] for _ in spot]  # Node:现场行动:多个节点

    level = graph.run("match ({{name:'{0}'}})-[r:种类]-({1}) return {1}".format(alarm_name, a))
    level_node = [_[a] for _ in level]  # Node:警情程度:多个节点
    treat_node = [level2treat(level_node[i]) for i in range(len(level_node))]  # Node:处置方式:多个节点
    return action_node, alarm_node, spot_node, level_node, treat_node


# action_node, alarm_node, spot_node, level_node, treat_node = input2related('长时间打架')
# print(action_node, alarm_node, spot_node, level_node, treat_node, sep='\n', end='\n')
