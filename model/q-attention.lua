require('nn.QLayer')
require('nn.ScalarMult')


MULTILAYER_ATTENTION = not true

local function createQAttentionReaderModule()
   attention_module_size = 200

   if not (params.TASK == 'combined-q') then
      crash()
   end
   local word_input = nn.Identity()()
   local xemb = nn.LookupTable(params.vocab_size,params.embeddings_dimensionality)(word_input)
   local prev_h = nn.Identity()()
   local prev_c = nn.Identity()()

   local inputNodes = {word_input, prev_c, prev_h}
   local attentionInputTable = {}
   local x2h
   local y2h
   local z2h
   if not attention.ABLATE_INPUT then
       x2h = nn.Linear(params.embeddings_dimensionality, attention_module_size)(nn.BlockGradientLayer(params.batch_size, params.embeddings_dimensionality)(xemb))
       table.insert(attentionInputTable, x2h)
   end
   if not attention.ABLATE_STATE then
       -- better do not backpropagate into the recurrent state?
--       local recurrent_att_input = prev_c
       local recurrent_att_input = nn.BlockGradientLayer(params.batch_size, params.rnn_size)(prev_c)
       y2h = nn.Linear(params.rnn_size, attention_module_size)(recurrent_att_input)
       table.insert(attentionInputTable, y2h)
   end
   if not attention.ABLATE_SURPRISAL then
       local surprisal = nn.Identity()()
       z2h = nn.Linear(1, attention_module_size)(surprisal)
       table.insert(attentionInputTable, z2h)
       table.insert(inputNodes, surprisal)
   end
   local hidden_attention
   if #attentionInputTable > 1 then
      hidden_attention = nn.Sigmoid()(nn.CAddTable()(attentionInputTable))
   else
      hidden_attention = nn.Sigmoid()(attentionInputTable[1])
   end

   if MULTILAYER_ATTENTION then
      hidden_attention = nn.Sigmoid()(nn.Linear(attention_module_size, attention_module_size)(hidden_attention))
   end

   -- the Q values for attending (2) and skipping (1)
   local attention = nn.Linear(attention_module_size, 2)(hidden_attention)
   local attention_decisions = nn.AddConstant(-1)(nn.QLayer(2)(nn.BlockGradientLayer(params.batch_size, 2)(attention)))
   local reader_input = nn.ScalarMult()({attention_decisions,xemb})
   -- normalizes innermost dimension
--   local normalized_input = nn.Normalize(2)(unnormalized_input)
   local next_c, next_h = lstm.lstm(reader_input, prev_c, prev_h, params.embeddings_dimensionality)
   local h2y              = nn.Linear(params.rnn_size, params.vocab_size)(next_c)
   local output           = nn.MulConstant(-1)(nn.LogSoftMax()(h2y))
   local module = nn.gModule(inputNodes,
                                      {next_c, next_h, output, attention, attention_decisions})
   module:getParameters():uniform(-params.init_weight, params.init_weight)
   return transfer_data(module)
end
-- other objective for economy




function setupCombinedQ()
   if not (params.TASK == 'combined-q') then
      crash()
   end
  print("Creating a RNN LSTM network.")

 -- initialize data structures
  -- TODO why is there no declaration for model.sA, model.sRA here?
  model.sR = {}
  model.dsR = {}
  model.dsA = {}
  model.start_sR = {}
  for j = 0, params.seq_length do
    model.sR[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

    model.dsR[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsR[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsR[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h
  model.dsR[4] = transfer_data(torch.zeros(params.batch_size, 2)) -- Q values
  model.dsR[5] = transfer_data(torch.zeros(params.batch_size, 1)) -- attention decisions



    model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h


-- global to make more efficient, but should not be referenced outside of bpCombinedQ
qtarget = torch.zeros(params.batch_size):cuda()




  reader_c ={}
  reader_h = {}

  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}

  reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() --TODO they have to be intiialized decently
  reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()


  reader_output = {}
  surprisal_values = {[1] = transfer_data(torch.zeros(params.batch_size,1))}



  target_criterion = nn.MSECriterion():cuda()



  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  for i=1, params.seq_length do
     attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
  end

  totalAttentions = torch.CudaTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  if params.TASK == 'combined-q' then
      nll_reader = torch.FloatTensor(params.batch_size)
  end





  ones = torch.ones(params.batch_size):cuda()
  rewardBaseline = 0

   variance_average = 100
   recurrent_variance_average = 100
   if not LOAD then
     -- READER
     local reader_core_network
     reader_core_network = createQAttentionReaderModule()
     paramxR, paramdxR = reader_core_network:getParameters()

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end

  elseif true then

     print("LOADING MODEL AT ".."/disk/scratch2/s1582047/model-"..fileToBeLoaded)
-- TODO add params
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load("/disk/scratch2/s1582047/model-"..fileToBeLoaded, "binary"))



----      modelsArray = {params,(numberOfWords/params.seq_length),uR, udR, uA, udA, reader_c[0], reader_h[0]}


    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)


------
     local reader_core_network
     reader_core_network = createQAttentionReaderModule()


     local reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end



     paramxR, paramdxR = reader_core_network:getParameters()

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

------
     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j])
           actor_network_gradparams[j]:set(SparamdxA[j])
     end


     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end


     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end
end





function fpCombinedQ(corpus, startIndex, endIndex)
   if not (params.TASK == 'combined-q') then
      crash()
   end

  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE) --hullu
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  if params.TASK == 'combined-q' then
     nll_reader:zero()
     reader_output = {}
  end


  for i=1, params.seq_length do
     local inputTensor = inputTensors[i]
          if i>1 then
             surprisal_values[i] = retrieveSurprisalValue(reader_output[i-1], inputTensors[i])
          end

          local inputs = {inputTensor, reader_c[i-1], reader_h[i-1]}
          if not attention.ABLATE_SURPRISAL then
            table.insert(inputs, surprisal_values[i])
          end

-- where attention_scores[i] is the two Q values of the possible actions in step i
          reader_c[i], reader_h[i], reader_output[i], attention_scores[i], attention_decisions[i] = unpack(readerRNNs[i]:forward(inputs))
          totalAttentions:add(attention_decisions[i])
--          print(261)
  --        print(i)
    --      print(totalAttentions)
      --    print(attention_decisions[i])
          if i < params.seq_length then
             for item=1, params.batch_size do
                local lm_loss_for_item =  reader_output[i][item][getFromData(corpus,startIndex+ item - 1,i+1)] 
                nll_reader[item] = nll_reader[item] + lm_loss_for_item
             end
          end
  end
--  print("261")
  --print(totalAttentions)

  actor_c[0] = reader_c[params.seq_length] 
  actor_h[0] = reader_h[params.seq_length] 


  nll:zero()
  actor_output = {}
  
  local inputTensor
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({inputTensor, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        local rec_loss_for_item =  actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)] 
        nll[item] = nll[item] + rec_loss_for_item
     end
  end

  return nll, actor_output
end



function bpCombinedQ(corpus, startIndex, endIndex)
   if not (params.TASK == 'combined-q') then
      crash()
   end
  
  paramdxR:zero()
  paramdxA:zero()
  
  
  
  reset_ds()
  
  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)
  
  
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 then
   -- do it for actor network
    if TRAIN_AUTOENCODER then
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
  

          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()
          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          cutorch.synchronize()
      end
  
      model.dsR[1]:copy(model.dsA[1])
      model.dsR[2]:copy(model.dsA[2])
      model.dsR[3]:zero()
   else
      model.dsR[1]:zero()
      model.dsR[2]:zero()
      model.dsR[3]:zero()
   end
   model.dsR[5]:zero() 


   actionIndices = torch.CudaTensor(params.batch_size, 1)

      -- TODO first c, h are not trained
      -- do it for reader network
      --
      --

    -- the Q value predicted at i+1
    local predictedQValueLast = nil

    for i = params.seq_length, 1, -1 do
     
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])
  
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
 
 --         print(356)
   --       print(qtarget)
          -- error of Q values.
          if i == params.seq_length then
             torch.add(qtarget, nll:cuda(), params.TOTAL_ATTENTIONS_WEIGHT, totalAttentions):add(nll_reader:cuda())
             qtarget:mul(-1)
         else
             qtarget:copy(predictedQValueLast)
          end
          model.dsR[4]:zero()
          actionIndices:copy(attention_decisions[i]):add(1)
    --      print(actionIndices)

          -- the Q value predicted at i
          local predictedQValues = attention_scores[i]:gather(2, actionIndices)
     --     print({predictedQValues, qtarget:view(-1,1)})
          local targetGradient = target_criterion:backward(predictedQValues, qtarget:view(-1,1))

          -- make learning slower for the Q approximation
          targetGradient:mul(0.2)
          model.dsR[4]:scatter(2, actionIndices, targetGradient)




--          print(model.dsR[4])
  --        print(379)
          predictedQValueLast = predictedQValues

          local tmp = readerRNNs[i]:backward({inputTensor, reader_c[i-1], reader_h[i-1], surprisal_values[i]},
                                        model.dsR)
--dsR:
--1 reader_c
--2 reader_h
--3 reader_output[i]
--4 attention_scores[i], i.e. Q values
--5 attention_decisions[i]


--print({reader_c[i], reader_h[i], reader_output[i], attention_scores[i], attention_decisions[i]})
--print(model.dsR)

          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          model.dsR[3]:zero()
          if TRAIN_LANGMOD then
             buildGradientsOfProbOutputs(model.dsR[3], corpus, startIndex, endIndex, i)
          end
--          model.dsR[4]:fill(params.TOTAL_ATTENTIONS_WEIGHT)


          cutorch.synchronize()
      end
  
      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end
 

if model.norm_dwR > 10000 then
 print(model.norm_dwR)
end
 
      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end
 
      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))
      --print(paramdxR:mul(-params.lr):mean()) 
  end
  
  
  
end




function printStuffForCombinedQ(perp, actor_output, since_beginning, epoch, numberOfWords)
            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(experimentNameOut)
            print(torch.sum(nll_reader)/params.batch_size)
            print(torch.sum(nll)/params.batch_size)
            --print(nll:add(-1,nll_reader))
            
            print(params) 
   
            for l = 1, 1 do
               print("....")
               print(perp[l])
               for j=2,params.seq_length do
                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
                  local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
                  io.write((chars[getFromData(corpus,l,j)]))--..'\n')
                  io.write(" \t "..chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(corpus,l,j)]).." \t "..chars[predictedTokensLM[1]].."  "..math.exp(-predictedScoresLM[1]).."  "..math.exp(-reader_output[j-1][l][getFromData(corpus,l,j)]).." \t "..attention_decisions[j][l][1].."  "..attention_scores[j][l][1].."  "..attention_scores[j][l][2].."\n")
               end



print("REWARD "..(-(nll[l] + params.TOTAL_ATTENTIONS_WEIGHT * totalAttentions[l] + nll_reader[l])))


            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)
end


