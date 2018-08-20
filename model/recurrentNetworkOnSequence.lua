

recurrentNetworkOnSequence = {}
recurrentNetworkOnSequence.__name = "recurrentNetworkOnSequence"

recurrentNetworkOnSequence.number_of_LSTM_layers = 1
--recurrentNetworkOnSequence.ALSO_DO_LANGUAGE_MODELING = false --true
recurrentNetworkOnSequence.ACCESS_MEMORY = false
recurrentNetworkOnSequence.INITIALIZE_FROM_NEAT = true

print(recurrentNetworkOnSequence)


RecurrentNetworkOnSequence = torch.class('RecurrentNetworkOnSequence')

--require('nn.RecursorMod')
--require('nn.SequencerMod')
--require('nn.PrintLayer')
--require('nn.BlockGradientLayer')
--require('qaAttentionAnswerer')



function RecurrentNetworkOnSequence:__init(rnn_size, parameters, gradParameters, seq_length, doLanguageModeling)
  print("Creating a RNN LSTM network.")

 

  self.ALSO_DO_LANGUAGE_MODELING = doLanguageModeling or false

  assert(neatQA.DO_BIDIRECTIONAL_MEMORY)
  self.rnn_size = rnn_size
  self.dsR = {}
  self.states_cell ={}
  self.states_hidden = {}

  self.reader_output = {}

    self.states_cell[0] = torch.CudaTensor(params.batch_size,self.rnn_size):zero() 
    self.states_hidden[0] = torch.CudaTensor(params.batch_size,self.rnn_size):zero()


     local core_network = autoencoding.create_network(self.ALSO_DO_LANGUAGE_MODELING, true, true)

     local reader_network_params, reader_network_gradparams = core_network:parameters()

     

     if parameters ~= nil then
       print("Taking parameters")
       print(parameters)
       print(reader_network_params)
       for j=1, #parameters do
           if parameters[j] ~= "NIL_ENTRY" then
              reader_network_params[j]:set(parameters[j])
           end
           if gradParameters[j] ~= nil then
             reader_network_gradparams[j]:set(gradParameters[j])
           end
       end
     end


     self.param, self.paramd = core_network:getParameters()

     self.recurrentModules = {}

     if seq_length == nil then
       seq_length = params.seq_length
     end
     auxiliary.buildClones(seq_length, self.recurrentModules, core_network)
--     for i=1,params.seq_length do
  --      self.recurrentModules[i] = g_clone(core_network)
--auxiliary.printMemory("270")

 --    end





     self.paramd:zero()

 





end

function RecurrentNetworkOnSequence:parameters()
  return self.recurrentModules[1]:parameters() 
end



function RecurrentNetworkOnSequence:emptyCHTables()
  local readerCInitial = self.states_cell[0]
  local readerHInitial = self.states_hidden[0]
  self.states_cell = {}
  self.states_hidden = {}
  self.states_cell[0] = readerCInitial
  self.states_hidden[0] = readerHInitial

  self.reader_output = {}
end

function RecurrentNetworkOnSequence:fp(inputTensors,maximalLengthOccurringInInput, attentionObjects)
  self:emptyCHTables()
  --local inputTensorsTables = auxiliary.toUnaryTables(inputTensors)
  print("  41  "..maximalLengthOccurringInInput)

  
--= {attentionNetworks = attentionNetworks,questionForward = question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], questionBackward=question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], decisions=attention_decisions,scores=attention_scores}

  if attentionObjects ~= nil and neatQA.ATTENTION_DOES_Q_ATTENTION then
    neatQA.questionEmbeddings = neatQA.questionEncoder:forward(attentionObjects.questionInputTensors)
              neatQA.condition_mask = torch.Tensor(params.batch_size)
if neatQA.CONDITION == "mixed" then
          neatQA.condition_mask = neatQA.condition_mask:random(0,1)
elseif neatQA.CONDITION == "preview" or  neatQA.CONDITION == "fullpreview"  then
          neatQA.condition_mask = neatQA.condition_mask:fill(1)
elseif neatQA.CONDITION == "nopreview" or  neatQA.CONDITION == "fullnopreview" then
          neatQA.condition_mask = neatQA.condition_mask:fill(0)
else
 assert(false)
    end

          neatQA.condition_mask = neatQA.condition_mask:cuda()

  end
if (DOING_EVALUATION_OUTPUT and CREATE_RECORDED_FILES) or (DOING_EVALUATION_OUTPUT and torch.uniform()>0.7)   or (neatQA.STORE_Q_ATTENTION and torch.uniform() > 0.99) then
  showQAttentionNow = true
else
  showQAttentionNow = false
end

       if attentionObjects ~= nil and showQAttentionNow then
--          for batch=1,params.batch_size do
             batch = 1
             for q=1,QUESTION_LENGTH do
               if (attentionObjects.questionInputTensors[q][batch] > 0) then
                  token = readDict.chars[attentionObjects.questionInputTensors[q][batch]]
               else
                  break
               end
               print("#  "..token)
             end
             if true or neatQA.CONDITION == "mixed" then
                print("Condition :  "..(({"No Preview", "Preview"})[neatQA.condition_mask[batch]+1]))
             end
             print("PRINTING Q ATTENTION")
             print("TOK           \tATT\tQ?\tINPUT\tQ_GATE\tPROD\tFUT_GA\tFUT\tHIST_GA\tFROM_HI\tP_GAT\tPOSIT\tCOND\tCO*PO\tLastF\tLF_G")


               local printNumbers = {globalForExpOutput.attRelativeToQ, globalForExpOutput.fromInput, globalForExpOutput.gateFromInput, globalForExpOutput.dotproductAffine, globalForExpOutput.questHistoryFutureGate, globalForExpOutput.gatedQuestForFuture, globalForExpOutput.questHistoryOutGate, globalForExpOutput.gatedFromHistory, globalForExpOutput.positionGate, globalForExpOutput.positionGated }
               if true or neatQA.CONDITION == "mixed" then
                  table.insert(printNumbers, globalForExpOutput.conditionGate)
                  table.insert(printNumbers, globalForExpOutput.conditionTimesPositionGate)
               end
               if globalForExpOutput.lastFixHistory.output ~= nil then
                 table.insert(printNumbers, globalForExpOutput.lastFixHistory )
                 table.insert(printNumbers, globalForExpOutput.gateLastFixHistory)
               end
               local printString = "MEAN          \t  "
               for i=1,#printNumbers do
                    --print(printNumbers[i])
                    if printNumbers[i].mean == nil then
                       printString = printString.."\t".."NIL"
                    else
                       printString = printString.."\t"..string.format("%.4f",printNumbers[i].mean)
                    end
               end
               print(printString)

  --        end
       end



  for i=1, math.min(#self.recurrentModules, maximalLengthOccurringInInput) do
     -- run attention network
     if attentionObjects ~= nil then
       local attentionNetwork = attentionObjects.attentionNetworks[i]
       local originalInputTensor = attentionObjects.originalInputTensors[i]
       local attentionArguments = {originalInputTensor}
       if true or neatQA.CONDITION == "preview" then
         if neatQA.ATTENTION_DOES_Q_ATTENTION then
           table.insert(attentionArguments, neatQA.questionEmbeddings)
         elseif params.useBackwardQForAttention then
           assert(false)
           assert(not(neatQA.USE_INNOVATIVE_ATTENTION))
           table.insert(attentionArguments, attentionObjects.questionBackward)
         else
           assert(false)
           table.insert(attentionArguments, attentionObjects.questionForward)
         end
         if neatQA.USE_INNOVATIVE_ATTENTION then
             assert(false)
             table.insert(attentionArguments, attentionObjects.questionBackward)
         end
       end
       assert(neatQA.questionHistory[i-1] ~= nil)
       table.insert(attentionArguments, self.states_hidden[i-1])
       table.insert(attentionArguments, self.states_cell[i-1])
       table.insert(attentionArguments, neatQA.questionHistory[i-1])
       table.insert(attentionArguments, neatQA.positionTensors[i])
       table.insert(attentionArguments, neatQA.questionHistoryFromSource[i-1])
       if true or neatQA.CONDITION == "mixed" then
          table.insert(attentionArguments, neatQA.condition_mask)
       end
       local blockedAttention, blockedDecisions, probs, blockedInput,  qHist, qHistFS = unpack(attentionNetwork:forward(attentionArguments))
--       print(qHist)
       assert(qHist ~= nil)
       neatQA.questionHistory[i] = qHist
       neatQA.questionHistoryFromSource[i] =  qHistFS

       if CREATE_RECORDED_FILES then
         for name, predictor in pairs(globalForExpOutput) do
           if type(predictor) == "table" and predictor.output ~= nil  then
             if (predictor.outputRecord == nil) then
                predictor.outputRecord = {}
             end
             --print(name)
             --print(predictor)
             predictor.outputRecord[i] = predictor.output:float()
           end
         end
       end


       if showQAttentionNow then
             batch=1
             if attentionObjects.originalInputTensors[i][batch] ~= 0 then
               assert(globalForExpOutput.attRelativeToQ.output[batch][1] ~= nil)
               assert(globalForExpOutput.gateFromInput.output[batch][1] ~= nil)

               local fromInput = ""
               if (globalForExpOutput.fromInput.output ~= nil) then
                   fromInput = globalForExpOutput.fromInput.output[batch][1]
                   gate = globalForExpOutput.gateFromInput.output[batch][1]
                   prod = globalForExpOutput.dotproductAffine.output[batch][1]
               end
               local printNumbers = {blockedAttention[batch][1], globalForExpOutput.attRelativeToQ.output[batch][1], fromInput, gate, prod, globalForExpOutput.questHistoryFutureGate.output[batch][1], globalForExpOutput.gatedQuestForFuture.output[batch][1], globalForExpOutput.questHistoryOutGate.output[batch][1], globalForExpOutput.gatedFromHistory.output[batch][1], globalForExpOutput.positionGate.output[batch][1], globalForExpOutput.positionGated.output[batch][1]}
               if true or neatQA.CONDITION == "mixed" then
                  table.insert(printNumbers, globalForExpOutput.conditionGate.output[batch][1])
                  table.insert(printNumbers, globalForExpOutput.conditionTimesPositionGate.output[batch][1])
               end
               if globalForExpOutput.lastFixHistory.output ~= nil then
                 table.insert(printNumbers, globalForExpOutput.lastFixHistory.output[batch][1])
                 table.insert(printNumbers, globalForExpOutput.gateLastFixHistory.output[batch][1])
               end

               local printString = (readDict.chars[attentionObjects.originalInputTensors[i][batch]].."           "):sub(1,10)
               for i=1,#printNumbers do
                  printString = printString.."\t"..string.format("%.4f",printNumbers[i])
               end
               print(printString)
             end
       end



       attentionObjects.decisions[i]:copy(blockedDecisions)
       attentionObjects.scores[i]:copy(blockedAttention)
       attentionObjects.probabilities[i]:copy(probs)



       if false then
--          for batch=1,params.batch_size do
             batch=1
             print(readDict.chars[attentionObjects.originalInputTensors[i][batch]].."\t"..blockedAttention[batch][1])
             for q=1,QUESTION_LENGTH do
               if (attentionObjects.questionInputTensors[q][batch] > 0) then
                  token = readDict.chars[attentionObjects.questionInputTensors[q][batch]]
               else
                  break
               end
--               print(qProducts:size())
  --             print(attentionObjects.questionInputTensors[q][batch])
  --             print(attentionObjects.questionInputTensors[q]:size())
               print("#  "..token.."\t"..globalForExpOutput.attRelativeToQ.output[batch][1])
             end
             --print(qMax[batch][1])
             
  --        end
       end

--       attentionObjects.decisions[i] = attentionObjects.decisions[i]:view(params.batch_size,1)
  --     attentionObjects.scores[i] = attentionObjects.scores[i]:view(params.batch_size,1)
       inputTensors[i] = blockedInput
--       print(self.attentionObjects.decisions)
  --     print(self.attentionObjects.scores)
--       print("13316")
--       print(blockedDecisions)
  --     print(blockedAttention)
    --   print(probs)
      -- print(blockedInput)
     end


     -- run reader
     self.states_cell[i], self.states_hidden[i], self.reader_output[i] = unpack(self.recurrentModules[i]:forward({  inputTensors[i]  , self.states_cell[i-1], self.states_hidden[i-1]}))
  end
  return self.states_cell, self.states_hidden, self.reader_output
end

-- actorGradient:
-- 1 cell of last step
-- 2 hidden of last step
-- 3 cells over all time steps
-- 4 hidden over all time steps
function RecurrentNetworkOnSequence:bp(inputTensors, maximalLengthOccurringInInput, actorGradient, attentionObjects)


--attentionObjects = {attentionNetworks = attentionNetworks,questionForward = question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], questionBackward=question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], decisions=attention_decisions,scores=attention_scores, originalInputTensors = neatQA.inputTensors}

local attentionGradients
if attentionObjects ~= nil then
  attentionObjects.forwardHidden = self.states_hidden
  attentionObjects.forwardCell = self.states_cell
  attentionGradients = neatQA.doBackwardForAttention(attentionObjects) 
  -- keys: questionForward, readerState (indexed starting from 0)
else
  attentionGradients = nil
end



if false then
assert(false)
    print("13913")
    print(actorGradient)
end
    self.dsR[1] = transfer_data(torch.zeros(params.batch_size,self.rnn_size))
    self.dsR[2] = transfer_data(torch.zeros(params.batch_size,self.rnn_size)) --h
    if self.ALSO_DO_LANGUAGE_MODELING then
      self.dsR[3] = transfer_data(torch.zeros(params.batch_size,params.vocab_size)) --output
    end
 
if not true then
assert(false)
 print("Setting to zero 12927")
 self.paramd:zero()
end

  if params.lr > 0 then --hrhr
      self.paramd:mul(params.lr_momentum / (1-params.lr_momentum)) --better to put this inside this if condition
      if actorGradient[1] ~=nil then
        self.dsR[1]:copy(actorGradient[1])
      else
        self.dsR[1]:zero()
      end

      if actorGradient[2] ~=nil then
        self.dsR[2]:copy(actorGradient[2])
      else
        self.dsR[2]:zero()
      end

if false then
      assert(false)
      print("158152")
      print(self.paramd:norm())
end

      for i = math.min(#self.recurrentModules, maximalLengthOccurringInInput), 1, -1 do
if false then
  assert(false)
          print("16919")
end
          if actorGradient[3] ~= nil then 
             self.dsR[1]:add(actorGradient[3][i])
if false then
  assert(false)
             print(actorGradient[3][i]:norm())
end
          end 

          if actorGradient[4] ~= nil then 
             self.dsR[2]:add(actorGradient[4][i])
if false then
  assert(false)
             print(actorGradient[4][i]:norm())
end
          end 

if false then
  assert(false)
print("1768")
print(self.dsR)
print(self.dsR[1])
print(self.dsR[2])
end

--print("DO GRADIENT CHECK")
--assert(false)

          local prior_c = self.states_cell[i-1]
          local prior_h = self.states_hidden[i-1]
if false then
  assert(false)
          print("15819 "..i.."  "..self.dsR[1]:norm().."  "..self.dsR[2]:norm())
end
          local tmp = self.recurrentModules[i]:backward({inputTensors[i], prior_c, prior_h},
                                        self.dsR)
          self.dsR[1]:copy(tmp[2])
          self.dsR[2]:copy(tmp[3])
          cutorch.synchronize()

if false then
  assert(false)
  local p0, p0d = self.recurrentModules[1]:parameters()

  local p, pd = self.recurrentModules[i]:parameters()
  for t = 1, #p0 do
    assert(p0[t]:storage() == p[t]:storage())
    assert(p0d[t]:storage() == pd[t]:storage())
    assert(p0[t]:storage() == self.param:storage())
    assert(p0d[t]:storage() == self.paramd:storage())
print("OK")
  end
end


      end

if false then
  assert(false)
      print("15815")
      print(self.paramd:norm())
end

      auxiliary.clipGradients(self.paramd)
      auxiliary.updateParametersWithMomentum(self.param,self.paramd)
  end
--  recurrentNetworkOnSequence.counter = recurrentNetworkOnSequence.counter+1
--  if recurrentNetworkOnSequence.counter == 1000 then
--  crash()  
--  end
return attentionGradients
end
--recurrentNetworkOnSequence.counter = 0
