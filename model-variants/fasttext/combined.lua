combined = {}

function combined.bpCombined(corpus, startIndex, endIndex)
crash()

  paramdxR:zero()
  paramdxA:zero()

  -- MOMENTUM
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))


  reset_ds()
  --print(model.dsA)


  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)

    --[[print("***********")
   print(startIndex)
   print(endIndex)
   print(model.dsA[3])]]

  --model.dsA[3][1][corpus[startIndex][params.seq_length]] = 1 --TODO
  --model.dsA[3][2][corpus[startIndex+1][params.seq_length]] = 1 --TODO


   --local loss, _ = fp(data)
   --print("LOSS2 "..loss)

  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true

  if params.lr > 0 and (true or train_autoencoding) then --hrhr
   -- do it for actor network
    if TRAIN_AUTOENCODER then
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
          local derr = transfer_data(torch.ones(1))

   --print(model.dsA[1])
   --print(model.dsA[2])
   --print(model.dsA[3])

          --inputTensor = torch.CudaTensor(params.batch_size):zero() -- hullu
          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          --print(tmp)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()



          --model.dsA[3][1][corpus[startIndex][i-1]] = 1 --TODO
          --model.dsA[3][2][corpus[startIndex+1][i-1]] = 1 --TODO

          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          --print(model.dsA[1])
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

      -- TODO first c, h are not trained
      -- do it for reader network
      for i = params.seq_length, 1, -1 do
    
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])

          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
 
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
   --print(model.dsR[1])
   --print(model.dsR[2])
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          model.dsR[3]:zero()
          if TRAIN_LANGMOD then
             buildGradientsOfProbOutputs(model.dsR[3], corpus, startIndex, endIndex, i)
          end
          cutorch.synchronize()
          --print(paramdxR:norm())
      end

      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end

      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end

  --print(paramdxR)
      momentum = 0.8

      -- MOMENTUM
      --[[updatesR:mul(momentum)
      updatesR:add(paramdxR:mul(1-momentum))

      updatesA:mul(momentum)
      updatesA:add(paramdxA:mul(1-momentum))

      paramxR:add(updatesR:mul(-params.lr))
      paramxA:add(updatesA:mul(-params.lr))]]

      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))
      --print(paramxR[1])
      --print(paramxA[1])

  end

  -- reward = totalAttentions + negloglikelihood
  -- TODO

  if params.lr_att > 0 and (true and train_attention_network) then
     local reward = torch.add(nll,params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions):add(nll_reader):cuda() -- gives the reward for each batch item
    -- local rewardDifference = reward:add(-rewardBaseline, ones)
     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
     --rewardDifference:mul(REWARD_DIFFERENCE_SCALING)
     for i = params.seq_length, 1, -1 do


        if torch.uniform() > 0.9999 then
           print("REWARD")
           print(reward)
           print(baseline_scores[i])
           print(reward - baseline_scores[i])
           print("..2198")
        end
 

        local rewardDifference = baseline_scores[i]:add(-1, reward):mul(-1)

   

        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] =  params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  whatToMultiplyToTheFinalDerivative:cmul(rewardDifference)
        factorsForTheDerivatives:add(attentionEntropyFactor)

  --      print(baseline_scores[i])
--        print(reward)
        local d_baseline_error = rewardDifference:mul(-1):view(-1,1):mul(1)
--        print(factorsForTheDerivatives)
        local tmp = attentionNetworks[i]:backward({inputTensors[i], reader_c[i-1]},{factorsForTheDerivatives:mul(params.lr_att), d_baseline_error:mul(0.1)})
     end

     local norm_dwRA = paramdxRA:norm()
     if norm_dwRA > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwRA
        paramdxRA:mul(shrink_factor)
     end
     assert(norm_dwRA == norm_dwRA)
     print(paramdxRA:mean())
     -- MOMENTUM
     paramdxRA:mul((1-params.lr_momentum))
     paramxRA:add(paramdxRA:mul(- 1))
     paramdxRA:mul(-1) -- is this really better than cloning before multiplying?
  end


end

function combined.bpCombinedCollectiveBaseline(corpus, startIndex, endIndex)
crash()

  paramdxR:zero()
  paramdxA:zero()

  -- MOMENTUM
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))


  reset_ds()
  --print(model.dsA)


  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)

    --[[print("***********")
   print(startIndex)
   print(endIndex)
   print(model.dsA[3])]]

  --model.dsA[3][1][corpus[startIndex][params.seq_length]] = 1 --TODO
  --model.dsA[3][2][corpus[startIndex+1][params.seq_length]] = 1 --TODO


   --local loss, _ = fp(data)
   --print("LOSS2 "..loss)

  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true

  if params.lr > 0 and (true or train_autoencoding) then --hrhr
   -- do it for actor network
    if TRAIN_AUTOENCODER then
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
          local derr = transfer_data(torch.ones(1))

   --print(model.dsA[1])
   --print(model.dsA[2])
   --print(model.dsA[3])

          --inputTensor = torch.CudaTensor(params.batch_size):zero() -- hullu
          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          --print(tmp)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()



          --model.dsA[3][1][corpus[startIndex][i-1]] = 1 --TODO
          --model.dsA[3][2][corpus[startIndex+1][i-1]] = 1 --TODO

          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          --print(model.dsA[1])
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

      -- TODO first c, h are not trained
      -- do it for reader network
      for i = params.seq_length, 1, -1 do
    
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])

          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
 
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
   --print(model.dsR[1])
   --print(model.dsR[2])
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          model.dsR[3]:zero()
          if TRAIN_LANGMOD then
             buildGradientsOfProbOutputs(model.dsR[3], corpus, startIndex, endIndex, i)
          end
          cutorch.synchronize()
          --print(paramdxR:norm())
      end

      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end

      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end

  --print(paramdxR)
      momentum = 0.8

      -- MOMENTUM
      --[[updatesR:mul(momentum)
      updatesR:add(paramdxR:mul(1-momentum))

      updatesA:mul(momentum)
      updatesA:add(paramdxA:mul(1-momentum))

      paramxR:add(updatesR:mul(-params.lr))
      paramxA:add(updatesA:mul(-params.lr))]]

      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))
      --print(paramxR[1])
      --print(paramxA[1])

  end

  -- reward = totalAttentions + negloglikelihood
  -- TODO

  if params.lr_att > 0 and (true and train_attention_network) then
     local reward = torch.add(nll,params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions):add(nll_reader):cuda() -- gives the reward for each batch item
    -- local rewardDifference = reward:add(-rewardBaseline, ones)
     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
     --rewardDifference:mul(REWARD_DIFFERENCE_SCALING)
     for i = params.seq_length, 1, -1 do

        local rewardDifference = baseline_scores[i]:add(-1, reward):mul(-1)

        if torch.uniform() > 0.9999 then
           print("REWARD")
           print(reward)
           print(baseline_scores[i])
           print(rewardDifference)
           print("..2198")
        end
    

        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] =  params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  whatToMultiplyToTheFinalDerivative:cmul(rewardDifference)
        factorsForTheDerivatives:add(attentionEntropyFactor)

  --      print(baseline_scores[i])
--        print(reward)
        local d_baseline_error = rewardDifference:mul(-1):view(-1,1):mul(1)
--        print(factorsForTheDerivatives)
        local tmp = attentionNetworks[i]:backward({inputTensors[i], reader_c[i-1]},{factorsForTheDerivatives:mul(params.lr_att), d_baseline_error:mul(0.1)})
     end

     local norm_dwRA = paramdxRA:norm()
     if norm_dwRA > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwRA
        paramdxRA:mul(shrink_factor)
     end
     assert(norm_dwRA == norm_dwRA)
     print(paramdxRA:mean())
     -- MOMENTUM
     paramdxRA:mul((1-params.lr_momentum))
     paramxRA:add(paramdxRA:mul(- 1))
     paramdxRA:mul(-1) -- is this really better than cloning before multiplying?
  end


end


function combined.bpCombinedNoBaselineNetwork(corpus, startIndex, endIndex)
  
  
  paramdxR:zero()
  paramdxA:zero()
  
  -- MOMENTUM
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))
  
  
  reset_ds()
  --print(model.dsA)
  
  
  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)
  
    --[[print("***********")
   print(startIndex)
   print(endIndex)
   print(model.dsA[3])]]
  
  --model.dsA[3][1][corpus[startIndex][params.seq_length]] = 1 --TODO
  --model.dsA[3][2][corpus[startIndex+1][params.seq_length]] = 1 --TODO
  
  
   --local loss, _ = fp(data)
   --print("LOSS2 "..loss)
  
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 and (true or train_autoencoding) then --hrhr
   -- do it for actor network
    if TRAIN_AUTOENCODER then
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
          local derr = transfer_data(torch.ones(1))
  
   --print(model.dsA[1])
   --print(model.dsA[2])
   --print(model.dsA[3])
  
          --inputTensor = torch.CudaTensor(params.batch_size):zero() -- hullu
          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          --print(tmp)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()
  
  
  
          --model.dsA[3][1][corpus[startIndex][i-1]] = 1 --TODO
          --model.dsA[3][2][corpus[startIndex+1][i-1]] = 1 --TODO
  
          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          --print(model.dsA[1])
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
  
      -- TODO first c, h are not trained
      -- do it for reader network
      for i = params.seq_length, 1, -1 do
     
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])
  
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
  
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
   --print(model.dsR[1])
   --print(model.dsR[2])
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          model.dsR[3]:zero()
          if TRAIN_LANGMOD then
             buildGradientsOfProbOutputs(model.dsR[3], corpus, startIndex, endIndex, i)
          end
          cutorch.synchronize()
          --print(paramdxR:norm())
      end
  
      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end
  
      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end
  
  --print(paramdxR)
      momentum = 0.8
  
      -- MOMENTUM
      --[[updatesR:mul(momentum)
      updatesR:add(paramdxR:mul(1-momentum))
  
      updatesA:mul(momentum)
      updatesA:add(paramdxA:mul(1-momentum))
  
      paramxR:add(updatesR:mul(-params.lr))
      paramxA:add(updatesA:mul(-params.lr))]]
  
      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))
      --print(paramxR[1])
      --print(paramxA[1])
  
  end
  
  -- reward = totalAttentions + negloglikelihood
  -- TODO
  
  if params.lr_att > 0 and (true and train_attention_network) then
     local reward = torch.add(nll,params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions):add(nll_reader):cuda() -- gives the reward for each batch item


     local rewardDifference = reward:clone():cuda():add(-rewardBaseline, ones) 

     local baselineScores
     if USE_BIDIR_BASELINE then
        bidir_baseline_gradparams:zero()
        local inputsWithActions = {}
        for i=1, params.seq_length do
            inputsWithActions[i] = joinTable[i]:forward({inputTensors[i]:view(-1,1), attention_decisions[i]:view(-1,1)})
        end
--        print(inputsWithActions[1])
  --      print(inputsWithActions[2])
        baselineScores = bidir_baseline:forward(inputsWithActions)
--        print(baselineScores)
        for i=1, params.seq_length do
          local target
          if i == params.seq_length then
             target = reward:view(-1,1)
             --target = totalAttentions:view(-1,1):cuda()
         else
             target = baselineScores[i+1]:view(-1,1)
          end

             if false and i == params.seq_length and torch.uniform() > 0.9 then
               print("TARGET ")
               print(target)

               print("SCORES ")
               print(baselineScores[i])
             end
 



          bidirBaseline.criterion_gradients[i]:copy(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          if i == params.seq_length then
             --print(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          end
       end
       bidir_baseline:backward(inputsWithActions, bidirBaseline.criterion_gradients)

     local norm_dwB = bidir_baseline_gradparams:norm()
     if norm_dwB > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwB
        bidir_baseline_gradparams:mul(shrink_factor)
     end
       bidir_baseline_params:add(bidir_baseline_gradparams:mul(- 0.1))
--       print(2769)
  --     print(bidir_baseline_gradparams:norm())

     end

     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
    if USE_BIDIR_BASELINE then
        local variance = rewardDifference:clone():cmul(rewardDifference):sqrt():mean()
        variance_average = 0.99 * variance_average + 0.01 * variance
if false then
  print("MEAN  "..variance_average)
  print("BIDIR "..recurrent_variance_average)
end

     end
     for i = params.seq_length, 1, -1 do
        rewardDifferenceForI = rewardDifference
        if USE_BIDIR_BASELINE then
            if i>1 then
              rewardDifferenceForI = baselineScores[i-1]:clone():add(-1,reward):mul(-1)
if torch.uniform() > 0.9 and i == 25 then
 print(i.."   "..(torch.mean(torch.abs(rewardDifference) - torch.abs(rewardDifferenceForI))))
end
               local variance = torch.cmul(rewardDifferenceForI, rewardDifferenceForI):sqrt():mean()
               recurrent_variance_average = 0.999 * recurrent_variance_average + 0.001 * variance

            end
        end

       

        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] =  params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  rewardDifferenceForI:clone():cmul(whatToMultiplyToTheFinalDerivative)
        factorsForTheDerivatives:add(attentionEntropyFactor)

     local inputTensor
     if i>1 and USE_PREDICTION_FOR_ATTENTION then
        inputTensor = attention_inputTensors[i]
     else 
         inputTensor = inputTensors[i-1]
     end


        local tmp = attentionNetworks[i]:backward({inputTensor, reader_c[i-1]},factorsForTheDerivatives)
     end
     local norm_dwRA = paramdxRA:norm()
     if norm_dwRA > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwRA
        paramdxRA:mul(shrink_factor)
     end
     assert(norm_dwRA == norm_dwRA)
  
     -- MOMENTUM
     paramdxRA:mul((1-params.lr_momentum))
     paramxRA:add(paramdxRA:mul(- 1 * params.lr_att))
     paramdxRA:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?
  end
  
  
end







function combined.printStuffForCombined(perp, actor_output, since_beginning, epoch, numberOfWords)
            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..readChunks.corpusReading.currentFile..
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
                  io.write((readDict.chars[getFromData(readChunks.corpus,l,j)]))--..'\n')
                  io.write(" \t "..readDict.chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(readChunks.corpus,l,j)]).." \t "..readDict.chars[predictedTokensLM[1]].."  "..math.exp(-predictedScoresLM[1]).."  "..math.exp(-reader_output[j-1][l][getFromData(readChunks.corpus,l,j)]).." \t "..attention_decisions[j][l].."  "..attention_scores[j][l][1].."\n")
               end
            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)


end







