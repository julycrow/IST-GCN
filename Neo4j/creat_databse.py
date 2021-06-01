# coding:utf-8
from py2neo import Graph, Node, Relationship, NodeSelector

graph = Graph('bolt://localhost:7687', username='neo4j', password='0000')


def alarm2level(graph, alarm, level):
    exist = graph.find_one(label='警情', property_key='name', property_value=alarm)
    if exist:
        n1 = exist
    else:
        n1 = Node('警情', name=alarm)
    exist = graph.find_one(label='警情程度', property_key='name', property_value=level)
    if exist:
        n2 = exist
    else:
        n2 = Node('警情程度', name=level)
    r = Relationship(n1, '种类', n2)
    s = n1 | n2 | r
    graph.create(s)


def level2treat(graph, level, treat):
    exist = graph.find_one(label='警情程度', property_key='name', property_value=level)
    if exist:
        n1 = exist
    else:
        n1 = Node('警情程度', name=level)
    exist = graph.find_one(label='处置方式', property_key='name', property_value=treat)
    if exist:
        n2 = exist
    else:
        n2 = Node('处置方式', name=treat)
    r = Relationship(n1, '采取', n2)
    s = n1 | n2 | r
    graph.create(s)


def treat2law(graph, treat, law):
    exist = graph.find_one(label='处置方式', property_key='name', property_value=treat)
    if exist:
        n1 = exist
    else:
        n1 = Node('处置方式', name=treat)
    exist = graph.find_one(label='法律法规', property_key='name', property_value=law)
    if exist:
        n2 = exist
    else:
        n2 = Node('法律法规', name=law)
    r = Relationship(n1, '来源', n2)
    s = n1 | n2 | r
    graph.create(s)


def alarm2level2treat(graph, alarm, level, treat):
    exist = graph.find_one(label='警情', property_key='name', property_value=alarm)
    if exist:
        n1 = exist
    else:
        n1 = Node('警情', name=alarm)
    exist = graph.find_one(label='警情程度', property_key='name', property_value=level)
    if exist:
        n2 = exist
    else:
        n2 = Node('警情程度', name=level)
    exist = graph.find_one(label='处置方式', property_key='name', property_value=treat)
    if exist:
        n3 = exist
    else:
        n3 = Node('处置方式', name=treat)
    r1 = Relationship(n1, '种类', n2)
    r2 = Relationship(n2, '采取', n3)
    s = n1 | n2 | n3 | r1 | r2
    graph.create(s)


def alarm2level2treat2law(graph, alarm, level, treat, law):
    exist = graph.find_one(label='警情', property_key='name', property_value=alarm)
    if exist:
        n1 = exist
    else:
        n1 = Node('警情', name=alarm)
    exist = graph.find_one(label='警情程度', property_key='name', property_value=level)
    if exist:
        n2 = exist
    else:
        n2 = Node('警情程度', name=level)
    exist = graph.find_one(label='处置方式', property_key='name', property_value=treat)
    if exist:
        n3 = exist
    else:
        n3 = Node('处置方式', name=treat)
    exist = graph.find_one(label='法律法规', property_key='name', property_value=law)
    if exist:
        n4 = exist
    else:
        n4 = Node('法律法规', name=law)
    r1 = Relationship(n1, '种类', n2)
    r2 = Relationship(n2, '采取', n3)
    r3 = Relationship(n3, '来源', n4)
    s = n1 | n2 | n3 | n4 | r1 | r2 | r3
    graph.create(s)


def alarm2input2action(graph, alarm, input, action):
    exist = graph.find_one(label='警情', property_key='name', property_value=alarm)
    if exist:
        n1 = exist
    else:
        n1 = Node('警情', name=alarm)
    exist = graph.find_one(label='输入', property_key='name', property_value=input)
    if exist:
        n2 = exist
    else:
        n2 = Node('输入', name=input)
    exist = graph.find_one(label='小车行动', property_key='name', property_value=action)
    if exist:
        n3 = exist
    else:
        n3 = Node('小车行动', name=action)
    r1 = Relationship(n1, '输入分类', n2)
    r2 = Relationship(n2, '采取', n3)
    r3 = Relationship(n3, '处理', n1)
    s = n1 | n2 | n3 | r1 | r2 | r3
    graph.create(s)


def alarm_action2spot(graph, alarm, action, spot):
    exist = graph.find_one(label='警情', property_key='name', property_value=alarm)
    if exist:
        n1 = exist
    else:
        n1 = Node('输入', name=alarm)
    exist = graph.find_one(label='现场行动', property_key='name', property_value=spot)
    if exist:
        n2 = exist
    else:
        n2 = Node('现场行动', name=spot)
    exist = graph.find_one(label='小车行动', property_key='name', property_value=action)
    if exist:
        n3 = exist
    else:
        n3 = Node('小车行动', name=action)
    r1 = Relationship(n1, '执行', n2)
    r2 = Relationship(n3, '执行', n2)
    graph.create(n1 | n2 | n3 | r1 | r2)


# 打架
'''打架'''
# 分类处置与法律依据
alarm2level2treat2law(graph, '打架', '结伙斗殴', '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较重的，处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第二十六条')
alarm2level2treat2law(graph, '打架', '追逐、拦截他人', '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较重的，处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第二十六条')
alarm2level2treat2law(graph, '打架', '强拿硬要或者任意损毁、占用公私财物', '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较重的，处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第二十六条')
alarm2level2treat2law(graph, '打架', '其他寻衅滋事行为', '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较重的，处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第二十六条')
alarm2level2treat2law(graph, '打架', '殴打他人的，或者故意伤害他人身体', '处五日以上十日以下拘留，并处二百元以上五百元以下罚款；情节较轻的，处五日以下拘留或者五百元以下罚款',
                      '《中华人民共和国治安管理处罚法》第四十三条')
alarm2level2treat2law(graph, '打架', '结伙殴打、伤害他人', '处十日以上十五日以下拘留，并处五百元以上一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第四十三条')
alarm2level2treat2law(graph, '打架', '殴打、伤害残疾人、孕妇、不满十四周岁的人或者六十周岁以上的人', '处十日以上十五日以下拘留，并处五百元以上一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第四十三条')
alarm2level2treat2law(graph, '打架', '多次殴打、伤害他人或者一次殴打、伤害多人', '处十日以上十五日以下拘留，并处五百元以上一千元以下罚款',
                      '《中华人民共和国治安管理处罚法》第四十三条')
alarm2level2treat2law(graph, '打架', '故意伤害他人身体', '处三年以下有期徒刑、拘役或者管制', '《中华人民共和国刑法》第二百三十四条')
alarm2level2treat2law(graph, '打架', '犯前款罪，致人重伤', '处三年以上十年以下有期徒刑', '《中华人民共和国刑法》第二百三十四条')
alarm2level2treat2law(graph, '打架', '致人死亡或者以特别残忍手段致人重伤造成严重残疾', '处十年以上有期徒刑、无期徒刑或者死刑', '《中华人民共和国刑法》第二百三十四条')
# 行为决策
alarm2input2action(graph, '打架', '短时间打架', '声光警告')
alarm2input2action(graph, '打架', '长时间打架', '通知警员')
alarm_action2spot(graph, '打架', '通知警员',
                  '继续打架斗殴的，且现有警力无法处置的，立即呼叫支援，在保证自身安全的前提下，可以口头警告、徒手制止、使用警械制止，符合条件的，可以使用武器。不能制止的，要做好取证工作。制止后，不构成犯罪的，可以进行当场调解，不能调解的带至公安机关处理(有伤者的，拨打120急救电话进行救治)。')
alarm_action2spot(graph, '打架', '通知警员', '已经结束的，不构成犯罪的，可以进行当场调解，不能调解的带至公安机关处理(有伤者的，拨打120急救电话进行救治)。')

'''非法聚集'''
# 分类处置与法律依据
alarm2level2treat2law(graph, '非法聚集', '聚众扰乱社会秩序，情节严重，致使工作、生产、营业和教学、科研、医疗无法进行，造成严重损失',
                      '对首要分子，处三年以上七年以下有期徒刑；对其他积极参加的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利', '《刑法》第二百九十条')
alarm2level2treat2law(graph, '非法聚集', '聚众冲击国家机关，致使国家机关工作无法进行，造成严重损失',
                      '对首要分子，处五年以上十年以下有期徒刑；对其他积极参加的，处五年以下有期徒刑、拘役、管制或者剥夺政治权利',
                      '《刑法》第二百九十条')
alarm2level2treat2law(graph, '非法聚集', '多次扰乱国家机关工作秩序，经行政处罚后仍不改正，造成严重后果', '处三年以下有期徒刑、拘役或者管制', '《刑法》第二百九十条')
alarm2level2treat2law(graph, '非法聚集', '多次组织、资助他人非法聚集，扰乱社会秩序', '情节严重的，依照前款的规定处罚', '《刑法》第二百九十条')
alarm2level2treat2law(graph, '非法聚集', '聚众扰乱车站、码头、民用航空站、商场、公园、影剧院、展览会、运动场或者其他公共场所秩序，聚众堵塞交通或者破坏交通秩序，抗拒、阻碍国家治安管理工作人员依法执行职务',
                      '情节严重的，对首要分子，处五年以下有期徒刑、拘役或者管制', '《刑法》第二百九十一条')
alarm2level2treat2law(graph, '非法聚集', '投放虚假的爆炸性、毒害性、放射性、传染病病原体等物质，或者编造爆炸威胁、生化威胁、放射威胁等恐怖信息，或者明知是编造的恐怖信息而故意传播，严重扰乱社会秩序',
                      '处五年以下有期徒刑、拘役或者管制；造成严重后果的，处五年以上有期徒刑', '《刑法》第二百九十一条')
alarm2level2treat2law(graph, '非法聚集', '编造虚假的险情、疫情、灾情、警情，在信息网络或者其他媒体上传播，或者明知是上述虚假信息，故意在信息网络或者其他媒体上传播，严重扰乱社会秩序',
                      '处三年以下有期徒刑、拘役或者管制；造成严重后果的，处三年以上七年以下有期徒刑', '《刑法》第二百九十一条')
alarm2level2treat2law(graph, '非法聚集', '举行集会、游行、示威，未依照法律规定申请或者申请未获许可，或者未按照主管机关许可的起止时间、地点、路线进行，又拒不服从解散命令，严重破坏社会秩序',
                      '对集会、游行、示威的负责人和直接责任人员，处五年以下有期徒刑、拘役、管制或者剥夺政治权利', '《刑法》第二百九十六条')
alarm2level2treat2law(graph, '非法聚集', '以暴力、威胁方法阻碍国家机关工作人员依法执行职务', '处三年以下有期徒刑、拘役、管制或者罚金', '《刑法》第二百七十七条')
alarm2level2treat2law(graph, '非法聚集', '以暴力、威胁方法阻碍全国人民代表大会和地方各级人民代表大会代表依法执行代表职务', '依照前款的规定处罚', '《刑法》第二百七十七条')
alarm2level2treat2law(graph, '非法聚集', '在自然灾害和突发事件中，以暴力、威胁方法阻碍红十字会工作人员依法履行职责', '依照第一款的规定处罚', '《刑法》第二百七十七条')
alarm2level2treat2law(graph, '非法聚集', '故意阻碍国家安全机关、公安机关依法执行国家安全工作任务，未使用暴力、威胁方法，造成严重后果', '依照第一款的规定处罚', '《刑法》第二百七十七条')
alarm2level2treat2law(graph, '非法聚集', '暴力袭击正在依法执行职务的人民警察，造成严重后果', '依照第一款的规定从重处罚', '《刑法》第二百七十七条')
alarm2level2treat2law(graph, '非法聚集', '煽动、策划非法集会、游行、示威，不听劝阻', '处十日以上十五日以下拘留', '《治安管理处罚法》第五十五条')
alarm2level2treat2law(graph, '非法聚集', '举行集会、游行、示威，未依照法律规定申请或者申请未获许可；或者未按照主管机关许可的目的、方式、标语、口号、起止时间、地点、路线进行，不听制止',
                      '对其负责人和直接责任人员处以警告或者十五日以下拘留', '《集会游行示威法》第二十八条')
alarm2level2treat2law(graph, '非法聚集', '拒不执行人民政府在紧急状态情况下依法发布的决定、命令', '处警告或者二百元以下罚款；情节严重的，处五日以上十日以下拘留，可以并处五百元以下罚款',
                      '《治安管理处罚法》第五十条')
alarm2level2treat2law(graph, '非法聚集', '阻碍国家机关工作人员依法执行职务', '处警告或者二百元以下罚款；情节严重的，处五日以上十日以下拘留，可以并处五百元以下罚款', '《治安管理处罚法》第五十条')
alarm2level2treat2law(graph, '非法聚集', '阻碍执行紧急任务的消防车、救护车、工程抢险车、警车等车辆通行', '处警告或者二百元以下罚款；情节严重的，处五日以上十日以下拘留，可以并处五百元以下罚款',
                      '《治安管理处罚法》第五十条')
alarm2level2treat2law(graph, '非法聚集', '强行冲闯公安机关设置的警戒带、警戒区', '处警告或者二百元以下罚款；情节严重的，处五日以上十日以下拘留，可以并处五百元以下罚款',
                      '《治安管理处罚法》第五十条')
alarm2level2treat2law(graph, '非法聚集',
                      '以堵塞、封闭道路、出入口；起哄、闹事、辱骂；占据办公室、车间以及其他工作场所；纠缠、阻挠其他员工上班等方式扰乱机关、团体、企业、事业单位的工作秩序；扰乱车站、港口、码头、机场、商场、公园、展览馆或者其他公共场所秩序；扰乱公共汽车、电车、火车、船舶、航空器或者其他交通工具上的秩序；非法拦截或者强登、扒乘机动车、船舶、航空器以及其他交通工具，影响交通工具正常行驶',
                      '处警告或者二百元以下罚款；情节较重的，处五日以上十日以下拘留，可以并处五百元以下罚款。聚众实施前款行为的，对首要分子处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《治安管理处罚法》第二十三条')
alarm2level2treat2law(graph, '非法聚集', '散布谣言，谎报险情、疫情、警情，投放虚假的爆炸性、毒害性、放射性、腐蚀性物质或者传染病病原体等危险物质；扬言实施放火、爆炸、投放危险物质等扰乱公共秩序',
                      '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较轻的，处五日以下拘留或者五百元以下罚款',
                      '《治安管理处罚法》第二十五条')


# 行为决策
alarm2input2action(graph, '非法聚集', '短时间非法聚集', '声光警告')
alarm2input2action(graph, '非法聚集', '长时间非法聚集', '通知警员')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '一是发布通告，宣传政策。在现场处置时，应及时做好说服教育工作。通过广播宣传有关法律、法规和政策，发布命令或通告，阐明观点、说明事件真相，揭露少数人的阴谋，指出聚集人群的违法性，责令聚集者限时离去')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '二是分离扼守，抵抗冲击。限时离去的通告或命令发布的同时，要安排警力将聚集人群同要保护的目标进行分隔，使之不能接近或脱离接触。当聚集人群开始对警戒线和重要目标进行冲击时，警戒民警要收缩到适当位置（一般为警戒线以内），组成人墙或使用盾牌阻隔人群和抵挡抛掷物品。此阶段，无防护民警要尽量避免与人群直接接触，对于没有破坏重要目标和危及人民群众和民警生命安全的，不得使用武器进行反击')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '三是开进支援，封锁包围。为震慑聚集人员，形成大兵压境之势，现场指挥员应选择适当时机调动警力列队进入现场，并对聚集人群实行封锁包围。开进时可以根据支援队伍的数量、装备及攻坚能力和紧急程度，选择大编队正面突进和多路迂回开进等方式，迅速到达指定位置，形成威慑')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '四是分割穿插，突击隔绝。处置民警组成战术队形，按照先后顺序穿插到聚集的人群当中，对聚集人群进行分割和隔绝，使之分离成相互难以呼应的若干小块。具体战术可采取线型平推、短促强力冲击和双向挤压等方法；也可采用一点楔入向两侧分开、多点向心或平行楔入多片隔离等方式，辅以有较强战斗力的小分队多向穿插，将人群分割成若干块，阻断和隔绝其继续串联和聚集')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '五是短促冲击，强行驱散。为尽快平息事态，现场处置警力要运用战术采取强制手段将闹事人群驱逐出所占区域。进行强行驱散时，务求审时度势，充分准备，坚决果断，一举成功。具体战术可以选择“围三缺一，一线平推”、“中间突破，两翼卷击”、“穿插楔入，首取要害”、“短促冲击，强力驱赶”等方法')
# alarm_action2spot(graph, '非法聚集', '通知警员',
#                   '六是强制带离，实施抓捕。事件聚集人数较少，警力相对充足或经强行驱散但仍有少部分人拒不离开现场时，可以实行强行带离。在强行带离过程中，对策划指挥闹事的为首人员和骨干分子应严密监控，一举抓获。也可以采取跟踪尾随、事后抓捕的方法。非紧急和特殊情况下，一般不在人群中进行抓捕，现场情况紧急需要抓捕的，应组织小分队楔入并实施强力抓捕，小分队要有较强的战斗力，可分为穿插、攻击、掩护和抓捕等几个组互相策应。')
alarm_action2spot(graph, '非法聚集', '通知警员', '人民警察现场负责人命令解散')
alarm_action2spot(graph, '非法聚集', '通知警员', '拒不解散的，人民警察现场负责人有权依照国家有关规定决定采取必要手段强行驱散，并对拒不服从的人员强行带离现场或者立即予以拘留')

'''摔倒'''
# 分类处置
alarm2level2treat(graph, '摔倒', '当事人意识清醒', '可询问摔倒的原因，然后给予帮助。如是心绞痛发作，。')
alarm2level2treat(graph, '摔倒', '当事人意识丧失、大动脉搏动消失', '应视为猝死,对发生猝死的患者应立即使其平卧在地面上，严禁搬动。同时，要马上对其实施心肺复苏术，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人腰痛怀疑有腰椎骨折', '应在该处用枕或卷折好的毛毯垫好，使脊柱避免屈曲压迫脊髓，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人股骨颈骨折', '在该处用枕或卷折好的毛毯垫好，使脊柱避免屈曲压迫脊髓，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人腰痛怀疑有腰椎骨折', '应用木板固定骨折部位。木板长度相当于腋下到足跟处，在腋下及木板端，包好衬垫物，在胸廓部及骨盆部作环形包扎两周，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人其他部位骨折', '可用两条木板夹住骨折，上、中、下三部位用绷带固定，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人头颅损伤有耳鼻出血', '不要用纱布、棉花、手帕去堵塞，否则可导致颅内高压，并继发感染，并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人有心脑血管疾病、糖尿病等慢性病',
                  '可协助其服下随身携带的急救药，尽可能避免搬动患者，更不能抱住患者的身体进行摇晃。若患者坐在地上尚未完全倒下，可搬来椅子将其支撑住，或直接上前将其扶住。 若患者已完全倒地，可将其缓缓调整到仰卧位，同时小心地将其头面部偏向一侧，以防止其呕吐物误入气管而发生窒息。并迅速拨打120急救电话。')
alarm2level2treat(graph, '摔倒', '当事人有创口', '应用洁净毛巾、布单把创口包好，再用夹板固定，送附近医院诊治。')
alarm2level2treat(graph, '摔倒', '当事人醉酒', '与其沟通，询问亲人联系方式，然后给予帮助')

# 行为决策
alarm2input2action(graph, '摔倒', '短时间摔倒', '通知警员')
alarm2input2action(graph, '摔倒', '长时间摔倒', '通知警员')

'''打砸'''
# 分类处置与法律依据
alarm2level2treat2law(graph, '打砸', '故意损毁公私财物', '处五日以上十日以下拘留，可以并处五百元以下罚款；情节较重的，处十日以上十五日以下拘留，可以并处一千元以下罚款',
                      '《治安管理处罚法》第四十九条')
alarm2level2treat2law(graph, '打砸', '故意毁灭或者损坏公私财物，数额较大或者有其他严重情节的行为', '处三年以下有期徒刑、拘役或者罚金',
                      '《国刑法》第二百七十五条')
alarm2level2treat2law(graph, '打砸', '故意毁灭或者损坏公私财物，数额巨大或者有其他特别严重情节的行为', '处三年以上七年以下有期徒刑',
                      '《国刑法》第二百七十五条')
alarm2level2treat2law(graph, '打砸', '由于犯罪行为而使被害人遭受经济损失', '对犯罪分子除了依法给予刑事处罚外，并应当根据情况，判处赔偿经济损失',
                      '《国刑法》第36条第1款')
# 行为决策
alarm2input2action(graph, '打砸', '短时间打砸', '声光警告')
alarm2input2action(graph, '打砸', '长时间打砸', '通知警员')
alarm_action2spot(graph, '打砸', '通知警员', '情节较轻的，可以进行批评教育，必要时给予治安处罚')
alarm_action2spot(graph, '打砸', '通知警员', '对首要分子依照刑法抢劫罪的规定追究刑事责任')
