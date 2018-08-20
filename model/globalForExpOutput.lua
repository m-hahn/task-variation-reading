assert(globalForExpOutput == nil)

globalForExpOutput = {}

globalForExpOutput.accuracy = 0.0

globalForExpOutput.softAttentionsContainer = {}
globalForExpOutput.attRelativeToQ = {name = "ATT_RELATIVE_TO_Q"}
globalForExpOutput.fromInput = {name = "fromInput"}
globalForExpOutput.gateFromInput = {name = "gateFromInput"}
globalForExpOutput.dotproductAffine  = {name = "dotproductAffine"}
globalForExpOutput.questHistoryFutureGate  = {name = "questHistoryFutureGate"}
globalForExpOutput.gatedQuestForFuture  = {name = "gatedQuestForFuture"}
globalForExpOutput.questHistoryOutGate  = {name = "questHistoryOutGate"}
globalForExpOutput.gatedFromHistory = {name = "gatedFromHistory"}

globalForExpOutput.positionGate = {name = "positionGate"}
globalForExpOutput.positionGated = {name = "positionGated"}
globalForExpOutput.lastWordOccursGate1 = {name = "lastWordOccursGate1"}
globalForExpOutput.conditionGate = {name = "conditionGate"}
globalForExpOutput.conditionTimesPositionGate = {name = "conditionTimesPositionGate"}

globalForExpOutput.lastFixHistory = {name = "lastFixHistory"}
globalForExpOutput.gateLastFixHistory = {name = "gateLastFixHistory"}

