qaAttentionAnswerer = {}
qaAttentionAnswerer.__name = "qaAttentionAnswerer"

require('nn.ReplicateDynamic')
require('nn.ReplicateAdd')
require('nn.StoreInContainerLayer')
require('nn.VerifyDimensionsLayer')

print(qaAttentionAnswerer)



function qaAttentionAnswerer.createAnswerNetworkWithMemoryAndQuestionAttention(param,gradparam)
  local prev_c_table = nn.Identity()()
  local backward_prev_c_table = nn.Identity()()

  local question_forward = nn.Identity()()
  local question_backward = nn.Identity()()

  prev_c_table2 = nn.Identity()(prev_c_table)
--  prev_c_table2 = nn.PrintLayer("prev_c_table")(prev_c_table)

  local prev_c_join = nn.JoinTable(2)(prev_c_table2)
  prev_c_join = nn.View(params.batch_size,-1,params.rnn_size)(prev_c_join)

--  prev_c_join = nn.PrintLayer("prev_c_join",1,true)(prev_c_join)

  local backward_prev_c_join = nn.JoinTable(2)(backward_prev_c_table)
  backward_prev_c_join = nn.View(params.batch_size,-1,params.rnn_size)(backward_prev_c_join)

--  backward_prev_c_join = nn.PrintLayer("backward_prev_c_join",1,true)(backward_prev_c_join)
  

  local prev_cs_join = nn.JoinTable(3)({prev_c_join,backward_prev_c_join})

--  prev_cs_join = nn.PrintLayer("prev_cs_join",1,true)(prev_cs_join)

  

  question2_forward = nn.Identity()(question_forward)
--  question2_forward = nn.VerifyDimensionsLayer({params.batch_size,params.rnn_size},"4084")(question2_forward)
  question2_backward = nn.Identity()(question_backward)
  --question2_backward = nn.VerifyDimensionsLayer({params.batch_size,params.rnn_size},"4286")(question2_backward)

--  question2 = nn.PrintLayer("question1",1,true)(question2) -- 60*256
--  prev_cs_join = nn.PrintLayer("prev_cs_join2")(prev_cs_join) --60*(2*10*256)
  prev_cs_join2 = nn.Identity()(prev_cs_join)
--  prev_cs_join2 = nn.VerifyDimensionsLayer({params.batch_size,-1,2*params.rnn_size},"4784")(prev_cs_join2)

--  prev_cs_join2 = nn.PrintLayer("prev_cs_join2b")(prev_cs_join) --FORWARD: 60,10,512. BACK: 60,2560


  local question2 = nn.JoinTable(2)({question2_forward,question2_backward})
--  question2 = nn.VerifyDimensionsLayer({params.batch_size,2*params.rnn_size},"5379")(question2)

--  question2 = nn.PrintLayer("question2",1,true)(question2) 

  local questionReplicated = nn.ReplicateDynamic()({prev_cs_join2,question2})
--  questionReplicated = nn.VerifyDimensionsLayer({params.batch_size,-1},"5790")(questionReplicated)

  questionReplicated = nn.View(-1,2*params.rnn_size)(questionReplicated)

--  questionReplicated = nn.PrintLayer("questionReplicated3",1,true)(questionReplicated) --60*(10*256)

  prev_cs_join2 = nn.View(-1,2*params.rnn_size)(prev_cs_join2)
--  prev_cs_join2 = nn.PrintLayer("prev_cs_join5",1,true)(prev_cs_join2) --okay



  local attention = nn.Bilinear(2*params.rnn_size,2*params.rnn_size,1)({prev_cs_join2,questionReplicated})
--  attention = nn.PrintLayer("attention5b",1,true)(attention)

  --attention = nn.Sigmoid()(attention)
  attention = nn.View(params.batch_size,-1)(attention) 
--  attention = nn.PrintLayer("attention",1.0,true)(attention)

--  attention = nn.PrintLayer("attention",1,true)(attention)
  local attention_score = nn.SoftMax()(attention)
--  attention_score = nn.PrintLayer("attention_score6",1,true)(attention_score)
  attention_score = nn.View(params.batch_size,1,-1)(attention_score)


--  attention_score = nn.PrintLayer("attention_score6b")(attention_score)
  attention_score = nn.StoreInContainerLayer(globalForExpOutput.softAttentionsContainer)(attention_score)

--  attention_score = nn.PrintLayer("attention_score",1.0,true)(attention_score)

--  attention_score = nn.VerifyDimensionsLayer({params.batch_size,1,-1},"attention_score")(attention_score)


  prev_cs_join = nn.View(params.batch_size, -1, 2*params.rnn_size)(prev_cs_join)
--  prev_c_join = nn.PrintLayer("prev_c_join7",1,true)(prev_c_join)

--  attention_score = nn.PrintLayer("attention_score8")(attention_score)

   
  local prev_c = nn.MM(false,false)({attention_score, prev_cs_join}) -- should be (batchsize x rnnsize)

--  prev_c = nn.VerifyDimensionsLayer({params.batch_size,1,params.rnn_size},"prev_c")(prev_c)


--  prev_c = nn.PrintLayer("prev_c9",1,true)(prev_c)
  prev_c = nn.View(params.batch_size, 2*params.rnn_size)(prev_c)
--  prev_c = nn.PrintLayer("prev_c10")(prev_c)

  local decisionLinear = nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS)(prev_c)
  local decision = nn.LogSoftMax()(decisionLinear)
--  decision = nn.PrintLayer("decision",1,false)(decision)
  
-- {reader_c, auxiliary.reverseTable(backward_cs, neatQA.maximalLengthOccurringInInput[1]), question_cs[neatQA.maximalLengthOccurringInInputQuestion[1]}
  local module = nn.gModule({prev_c_table, backward_prev_c_table, question_forward, question_backward},
                                      {decision})

  parameters, gradparameters = module:parameters()
  for i=1, #parameters do
    local epsilon = math.sqrt(6.0/torch.LongTensor(parameters[i]:size()):sum())
    parameters[i]:uniform(-epsilon, epsilon)
    gradparameters[i]:zero()
  end
  for i=1,#param do
    parameters[i]:copy(param[i])
    gradparameters[i]:copy(gradparam[i])
  end
  return transfer_data(module)
end



-- mostly from Jianpeng's LM
function qaAttentionAnswerer.createAnswerNetworkWithMemoryAttention()
  assert(false)
  local prev_c_table = nn.Identity()()
  local lastState = nn.Identity()()
  local lastH = nn.Identity()()

--  prev_c_table2 = nn.PrintLayer("prev_c_table")(prev_c_table)

  local prev_c_join = nn.JoinTable(2)(prev_c_table)
--  prev_c_join = nn.PrintLayer("prev_c_join")(prev_c_join)

  local attention = nn.Linear(params.rnn_size,params.rnn_size)(nn.View(-1,params.rnn_size)(prev_c_join))
--  attention = nn.PrintLayer("attention")(attention)
  attention = nn.View(params.batch_size,-1)(attention)
--  attention = nn.PrintLayer("attention")(attention)
--  lastState2 = nn.PrintLayer("lastState")(lastState)
  local attention_sum = nn.Tanh()(nn.ReplicateAdd()({attention, lastState}))
--  attention_sum = nn.PrintLayer("attention_sum")(attention_sum)
  attention_sum = nn.View(-1,params.rnn_size)(attention_sum)
--  attention_sum = nn.PrintLayer("attention_sum")(attention_sum)


  local attention_score = nn.Linear(params.rnn_size, 1)(attention_sum)
--  attention_score = nn.PrintLayer("attention_score")(attention_score)
  attention_score = nn.View(params.batch_size,-1)(attention_score)
--  attention_score = nn.PrintLayer("attention_score")(attention_score) 
  attention_score = nn.SoftMax(2)(attention_score)
--  attention_score = nn.PrintLayer("attention_score")(attention_score)
  attention_score = nn.View(params.batch_size, 1, -1)(attention_score)
--  attention_score = nn.PrintLayer("attention_score")(attention_score)

  attention_score = nn.StoreInContainerLayer(globalForExpOutput.softAttentionsContainer)(attention_score)

  prev_c_join = nn.View(params.batch_size, -1, params.rnn_size)(prev_c_join)
--  prev_c_join = nn.PrintLayer("prev_c_join")(prev_c_join)

  
  local prev_c = nn.View(params.batch_size, params.rnn_size)(nn.MM(false,false)({attention_score, prev_c_join}))
--  prev_c = nn.PrintLayer("prev_c")(prev_c)

-- as described in Chen et al 2016 would be sufficient to get softmax directly from attention
  local inputsToDecision = nn.JoinTable(2)({lastState,lastH,prev_c})  
--  inputsToDecision  = nn.PrintLayer("inputsToDecision")(inputsToDecision)
  local decisionLinear = nn.Linear(3*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS)(inputsToDecision)
--  decisionLinear = nn.PrintLayer("decisionLinear")(decisionLinear)
  local decision = nn.LogSoftMax()(decisionLinear)
--  decision = nn.PrintLayer("decision")(decision)

    local module = nn.gModule({lastState,lastH,prev_c_table},
                                      {decision})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end


