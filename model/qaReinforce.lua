-- extracted from neat-qa-Unrolled-Attention.lua (25/11/2016)

assert(adam == nil)

function neatQA.initializeAdam()

  if adam == nil then
    print("Initializing Adam")
    adam = {}
    adam.__name = "Adam"
    adam.movingAverageOfGradients = paramdxRA:clone():zero()
    adam.movingAverageOfSquareGradients = paramdxRA:clone():zero()
    adam.temp = paramdxRA:clone():zero()
--    adam.square_gradient = paramdxRA:clone():zero()
    paramdxRA:zero()
    adam.adam_step = 0
    print(adam)
  end
end

neatQA.CORRECT_SCALING_L1 = true
print("L1 scaling ?  "..tostring(neatQA.CORRECT_SCALING_L1))

neatQA.constrainLogLikelihoodLoss = false
neatQA.constrainLogLikelihoodLossToValue = -1.0
print("Constraining log likelihood loss? "..tostring(neatQA.constrainLogLikelihoodLoss))
if neatQA.constrainLogLikelihoodLoss then
   print("Constraining log likelihood loss to "..tostring(neatQA.constrainLogLikelihoodLossToValue))
end

neatQA.attentionRateNoPreview = 0.5
neatQA.attentionRatePreview = 0.5
neatQA.attentionRate = 0.5

neatQA.accuracyPreview = 0.5
neatQA.accuracyNoPreview = 0.5
neatQA.accuracy = 0.5


neatQA.attentionRateNoPreviewSum = 0.0
neatQA.attentionRatePreviewSum = 0.0
neatQA.attentionRateSum = 0.0

neatQA.accuracyPreviewSum = 0.0
neatQA.accuracyNoPreviewSum = 0.0
neatQA.accuracySum = 0.0

neatQA.numberOfExamples = 0.0

neatQA.nllSumPreview = 0.0
neatQA.nllSumNoPreview = 0.0
neatQA.examplesPreview = 0.0000001
neatQA.examplesNoPreview = 0.0000001


function neatQA.doBackwardForAttention(attentionObjects) 
 neatQA.numberOfExamples = neatQA.numberOfExamples + 1

 local gradientsAttention = {}

 local questionHistoryGradient = neatQA.historyGradientStart
 local questionHistoryFromSourceGradient = neatQA.historyFromSourceGradientStart

     local logLikelihoodLoss = transfer_data(nll)

if true then
---------------------
  local nllPreview = torch.cmul(logLikelihoodLoss,neatQA.condition_mask):sum()
  local previewFraction = neatQA.condition_mask:sum()
  local nopreviewFraction = params.batch_size - previewFraction
  neatQA.examplesPreview = neatQA.examplesPreview + previewFraction
  neatQA.examplesNoPreview = neatQA.examplesNoPreview + nopreviewFraction
  neatQA.nllSumPreview = neatQA.nllSumPreview + nllPreview
  neatQA.nllSumNoPreview = neatQA.nllSumNoPreview + (nll:sum() - nllPreview)

     print("Total NLLs")
     print("G No Preview "..neatQA.nllSumNoPreview/neatQA.examplesNoPreview)
     print("G Preview    "..neatQA.nllSumPreview/neatQA.examplesPreview)


end




if neatQA.USE_ATTENTION_NETWORK then


  if neatQA.USE_ADAM_FOR_REINFORCE then
    neatQA.initializeAdam()
  else 
    auxiliary.prepareMomentum(paramdxRA)
  end

  if params.lr_att > 0 and (true and train_attention_network) then
     -- compute reward
     local reward = torch.CudaTensor(params.batch_size,1):zero()
     if neatQA.USE_GOLD_LIKELIHOODS_AS_BASELINE then
        assert(false)
        reward:add(neatQA.baselineTensor) -- this will make it negative
        --print("37412")
        --print(reward)
     end
     local gradientWRTAttentionScore

     -------------------
     -- Minimizing fixations
     -------------------
     local AVERAGE_FIXATION_RATE_OVER_BATCH = true
     print("Averaging fixation rate over batch? "..tostring(AVERAGE_FIXATION_RATE_OVER_BATCH))
     if neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS and (not AVERAGE_FIXATION_RATE_OVER_BATCH) then
        print("Have not implemented masking in this part")
        assert(false)

        -- compute the expected fixation rates per condition, just for the sake of displaying it
        local attentionRate = 0.0
        local attentionRatePreview = 0.0
        local attentionSumsPerItem = torch.CudaTensor(params.batch_size,1):zero()

        for i=1,neatQA.maximalLengthOccurringInInput[1] do
           attentionRate = attentionRate + attention_scores[i]:mean()
           attentionSumsPerItem:add(attention_scores[i])
        end
        attentionRate = attentionRate / neatQA.maximalLengthOccurringInInput[1]

        attentionSumsPerItem:cmul(neatQA.condition_mask)
        local previewPercentage = neatQA.condition_mask:mean() + 0.00000001
        attentionRatePreview = (attentionSumsPerItem:sum() / neatQA.maximalLengthOccurringInInput[1]) / (previewPercentage * params.batch_size)
        local attentionRateNoPreview = (attentionRate - (previewPercentage * attentionRatePreview)) / (1- previewPercentage)

        neatQA.attentionRateNoPreview = 0.95 * neatQA.attentionRateNoPreview + 0.05 * attentionRateNoPreview
        neatQA.attentionRatePreview = 0.95 * neatQA.attentionRatePreview + 0.05 * attentionRatePreview
        neatQA.attentionRate = 0.95 * neatQA.attentionRate + 0.05 * attentionRate

        print("Attention Rates Per Condition 9646")
        print("No Preview "..neatQA.attentionRateNoPreview)
        print("Preview    "..neatQA.attentionRatePreview)
        print("Total      "..neatQA.attentionRate)

        neatQA.attentionRateNoPreviewSum = neatQA.attentionRateNoPreviewSum + attentionRateNoPreview
        neatQA.attentionRatePreviewSum = neatQA.attentionRatePreviewSum + attentionRatePreview
        neatQA.attentionRateSum = neatQA.attentionRateSum + attentionRate


        print("G No Preview "..neatQA.attentionRateNoPreviewSum/neatQA.numberOfExamples)
        print("G Preview    "..neatQA.attentionRatePreviewSum/neatQA.numberOfExamples)
        print("G Total      "..neatQA.attentionRateSum/neatQA.numberOfExamples)



        -- perform optimization
        gradientWRTAttentionScore = torch.CudaTensor(params.batch_size,1):zero()

        for i=1,neatQA.maximalLengthOccurringInInput[1] do
          gradientWRTAttentionScore:add(attention_scores[i])
        end

        if neatQA.scaleAttentionWithLength then
          gradientWRTAttentionScore:div( neatQA.maximalLengthOccurringInInput[1] )
        end

        fileStatsFixations:write((neatQA.sessionCounter*params.batch_size)..'\t'..(gradientWRTAttentionScore:mean())..'\n')
        fileStatsFixations:flush()

        if neatQA.CUT_OFF_BELOW_BASELINE then
          gradientWRTAttentionScore:add(params.ATTENTION_VALUES_BASELINE)
          gradientWRTAttentionScore = neatQA.reluLayer:forward(gradientWRTAttentionScore)
          if true and torch.uniform() > 0.98 then
            print("3412b: KL divergence gradient")
            print(gradientWRTAttentionScore)
          end
        end

        if neatQA.use_l1 then
           gradientWRTAttentionScore:sign()
        end


        gradientWRTAttentionScore:mul(params.TOTAL_ATTENTIONS_WEIGHT)

     elseif neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS and AVERAGE_FIXATION_RATE_OVER_BATCH then
        local attentionRate = 0.0
        local attentionRatePreview = 0.0
        local attentionSumsPerItem = torch.CudaTensor(params.batch_size,1):zero()

        local itemLengths = neatQA.maxLengthsPerItem:cuda()
        local mask = torch.CudaTensor(params.batch_size)

        local totalLengthOfItems = itemLengths:sum()
        --print("total length")
      --  print(totalLengthOfItems)

        for i=1,neatQA.maximalLengthOccurringInInput[1] do
           mask:copy(itemLengths)
           mask:csub(i-1)
           mask:sign()
           mask:cmax(0)
--           if torch.uniform() > 0.999 and i > 400 then
  --           print("PADDING MASK")
    --         print(i)
      --       print(itemLengths)
        --     print(mask)
          -- end
           local attentionScoresModified = torch.cmul(attention_scores[i], mask)
           attentionSumsPerItem:add(attentionScoresModified)
        end
        attentionRate = attentionSumsPerItem:sum() / totalLengthOfItems

        attentionSumsPerItem:cmul(neatQA.condition_mask)
        local previewPercentage = neatQA.condition_mask:mean() + 0.000001
        attentionRatePreview = (attentionSumsPerItem:sum() / neatQA.maximalLengthOccurringInInput[1]) / (previewPercentage * params.batch_size)
        local attentionRateNoPreview = (attentionRate - (previewPercentage * attentionRatePreview)) / (1- previewPercentage)

        neatQA.attentionRateNoPreview = 0.95 * neatQA.attentionRateNoPreview + 0.05 * attentionRateNoPreview
        neatQA.attentionRatePreview = 0.95 * neatQA.attentionRatePreview + 0.05 * attentionRatePreview
        neatQA.attentionRate = 0.95 * neatQA.attentionRate + 0.05 * attentionRate



        print("Attention Rates Per Condition 9646")
        print("No Preview "..neatQA.attentionRateNoPreview)
        print("Preview    "..neatQA.attentionRatePreview)
        print("Total      "..neatQA.attentionRate)

        neatQA.attentionRateNoPreviewSum = neatQA.attentionRateNoPreviewSum + attentionRateNoPreview
        neatQA.attentionRatePreviewSum = neatQA.attentionRatePreviewSum + attentionRatePreview
        neatQA.attentionRateSum = neatQA.attentionRateSum + attentionRate


        print("G No Preview "..neatQA.attentionRateNoPreviewSum/neatQA.numberOfExamples)
        print("G Preview    "..neatQA.attentionRatePreviewSum/neatQA.numberOfExamples)
        print("G Total      "..neatQA.attentionRateSum/neatQA.numberOfExamples)





        ------------------------------------------
        ------------------------------------------

        local attentionRateCutOff = attentionRate + params.ATTENTION_VALUES_BASELINE

        gradientWRTAttentionScore = torch.CudaTensor(params.batch_size,1):fill(attentionRateCutOff)

        if neatQA.use_l1 then
           gradientWRTAttentionScore:sign()
           if neatQA.CORRECT_SCALING_L1 then
              gradientWRTAttentionScore:cdiv(itemLengths)
           end
        end

        fileStatsFixations:write((neatQA.sessionCounter*params.batch_size)..'\t'..(attentionRate)..'\n')
        fileStatsFixations:flush()

        assert(neatQA.CUT_OFF_BELOW_BASELINE)

          if true and torch.uniform() > 0.98 then
            print("3412b: KL divergence gradient")
            print(gradientWRTAttentionScore)
          end

        gradientWRTAttentionScore:mul(params.TOTAL_ATTENTIONS_WEIGHT)

    else --minimize fixations via REINFORCE


        local itemLengths = neatQA.maxLengthsPerItem:cuda()


       for i=1,neatQA.maximalLengthOccurringInInput[1] do
           reward:add(attention_decisions[i])
       end

       -- now scale with the length of the sequences
       if neatQA.scaleAttentionWithLength then
         reward:cdiv(itemLengths) -- neatQA.maximalLengthOccurringInInput[1] )
       else
         assert(false)
       end

       fileStatsFixations:write((neatQA.sessionCounter*params.batch_size)..'\t'..(reward:mean())..'\n')
       fileStatsFixations:flush()



       if neatQA.CUT_OFF_BELOW_BASELINE then
         reward:add(params.ATTENTION_VALUES_BASELINE)
         reward = neatQA.reluLayer:forward(reward)
         if true and torch.uniform() > 0.98 then
           print("3412: REWARD")
           print(reward)
         end
       end
       reward:mul(params.TOTAL_ATTENTIONS_WEIGHT)








        local attentionRate = 0.0
        local attentionRatePreview = 0.0
        local attentionSumsPerItem = torch.CudaTensor(params.batch_size,1):zero()

        local mask = torch.CudaTensor(params.batch_size)

        local totalLengthOfItems = itemLengths:sum()
        --print("total length")
      --  print(totalLengthOfItems)

        for i=1,neatQA.maximalLengthOccurringInInput[1] do
           mask:copy(itemLengths)
           mask:csub(i-1)
           mask:sign()
           mask:cmax(0)
--           if torch.uniform() > 0.999 and i > 400 then
  --           print("PADDING MASK")
    --         print(i)
      --       print(itemLengths)
        --     print(mask)
          -- end
           local attentionScoresModified = torch.cmul(attention_scores[i], mask)
           attentionSumsPerItem:add(attentionScoresModified)
        end
        attentionRate = attentionSumsPerItem:sum() / totalLengthOfItems

        attentionSumsPerItem:cmul(neatQA.condition_mask)
        local previewPercentage = neatQA.condition_mask:mean() + 0.000001
        attentionRatePreview = (attentionSumsPerItem:sum() / neatQA.maximalLengthOccurringInInput[1]) / (previewPercentage * params.batch_size)
        local attentionRateNoPreview = (attentionRate - (previewPercentage * attentionRatePreview)) / (1- previewPercentage)

        neatQA.attentionRateNoPreview = 0.95 * neatQA.attentionRateNoPreview + 0.05 * attentionRateNoPreview
        neatQA.attentionRatePreview = 0.95 * neatQA.attentionRatePreview + 0.05 * attentionRatePreview
        neatQA.attentionRate = 0.95 * neatQA.attentionRate + 0.05 * attentionRate



        print("Attention Rates Per Condition 9646")
        print("No Preview "..neatQA.attentionRateNoPreview)
        print("Preview    "..neatQA.attentionRatePreview)
        print("Total      "..neatQA.attentionRate)

        neatQA.attentionRateNoPreviewSum = neatQA.attentionRateNoPreviewSum + attentionRateNoPreview
        neatQA.attentionRatePreviewSum = neatQA.attentionRatePreviewSum + attentionRatePreview
        neatQA.attentionRateSum = neatQA.attentionRateSum + attentionRate


        print("G No Preview "..neatQA.attentionRateNoPreviewSum/neatQA.numberOfExamples)
        print("G Preview    "..neatQA.attentionRatePreviewSum/neatQA.numberOfExamples)
        print("G Total      "..neatQA.attentionRateSum/neatQA.numberOfExamples)





        ------------------------------------------
        ------------------------------------------

--        local attentionRateCutOff = attentionRate + params.ATTENTION_VALUES_BASELINE

        --gradientWRTAttentionScore = torch.CudaTensor(params.batch_size,1):fill(attentionRateCutOff)

--        if neatQA.use_l1 then
  --         gradientWRTAttentionScore:sign()
    --       if neatQA.CORRECT_SCALING_L1 then
      --        gradientWRTAttentionScore:cdiv(itemLengths)
        --   end
        --end

        fileStatsFixations:write((neatQA.sessionCounter*params.batch_size)..'\t'..(attentionRate)..'\n')
        fileStatsFixations:flush()

 --       assert(neatQA.CUT_OFF_BELOW_BASELINE)

  --        if true and torch.uniform() > 0.98 then
    --        print("3412b: KL divergence gradient")
      --      print(gradientWRTAttentionScore)
        --  end

--        gradientWRTAttentionScore:mul(params.TOTAL_ATTENTIONS_WEIGHT)












    end
    -------------------
    -- Done minimizing fixations
    -------------------




-- COMPUTE ACCURACIES
--neatQA.condition_mask
if true then
     local correctTensor = torch.FloatTensor(params.batch_size):zero()
     for l=1,params.batch_size do
       local answerID = qa.getFromAnswer(readChunks.corpus,l,1)
       local predictedScore,predictedAnswer = torch.max(actor_output[l],1)
       if answerID == predictedAnswer[1] then
         correctTensor[l] = 1.0
       end
     end
  --   print(correctTensor)
--     print(neatQA.condition_mask:float())
     local accuracy = correctTensor:mean()
        correctTensor:cmul(neatQA.condition_mask:float())
        local previewPercentage = neatQA.condition_mask:mean() + 0.000001
        local accuracyPreview = (correctTensor:sum()) / (previewPercentage * params.batch_size)
        local accuracyNoPreview = (accuracy - (previewPercentage * accuracyPreview)) / (1.0 - previewPercentage)

        neatQA.accuracyNoPreview = 0.95 * neatQA.accuracyNoPreview + 0.05 * accuracyNoPreview
        neatQA.accuracyPreview = 0.95 * neatQA.accuracyPreview + 0.05 * accuracyPreview
        neatQA.accuracy = 0.95 * neatQA.accuracy + 0.05 * accuracy

        print("Accuracy Per Condition 149")
        print("No Preview "..neatQA.accuracyNoPreview)
        print("Preview    "..neatQA.accuracyPreview)
        print("Total      "..neatQA.accuracy)

        neatQA.accuracyNoPreviewSum =  neatQA.accuracyNoPreviewSum + accuracyNoPreview
        neatQA.accuracyPreviewSum =  neatQA.accuracyPreviewSum +  accuracyPreview
        neatQA.accuracySum =  neatQA.accuracySum +  accuracy

        print("G No Preview "..(neatQA.accuracyNoPreviewSum/neatQA.numberOfExamples))
        print("G Preview    "..(neatQA.accuracyPreviewSum/neatQA.numberOfExamples))
        print("G Total      "..(neatQA.accuracySum/neatQA.numberOfExamples))



end


if neatQA.rewardBasedOnLogLikeLoss then
     if neatQA.constrainLogLikelihoodLoss then
        print("31517  "..tostring(nll:mean()))
        if nll:mean() > -neatQA.constrainLogLikelihoodLossToValue then
            reward:add(logLikelihoodLoss)
        else
            reward:add(-neatQA.constrainLogLikelihoodLossToValue)
        end
--        assert(false)
--        logLikelihoodLoss:add(neatQA.constrainLogLikelihoodLossToValue)
  --      logLikelihoodLoss = neatQA.reluLayer:forward(logLikelihoodLoss)
    --    logLikelihoodLoss:add(-neatQA.constrainLogLikelihoodLossToValue)
     else
            reward:add(logLikelihoodLoss)
     end

     -- as of 21/11/2016, these numbers are positive
else
     local lossTensor = torch.FloatTensor(params.batch_size):zero()
     for l=1,params.batch_size do
       local answerID = qa.getFromAnswer(readChunks.corpus,l,1)
       local predictedScore,predictedAnswer = torch.max(actor_output[l],1)
       if answerID ~= predictedAnswer[1] then
         lossTensor[l] = 1
       end
     end
    reward:add(transfer_data(lossTensor))
end

if torch.uniform()>0.999 then
 print("----38714")
 print(reward)
end

fileStatsReward:write((neatQA.sessionCounter*params.batch_size)..'\t'..(reward:mean())..'\n')
fileStatsReward:flush()



-- as of 19/11/2016: higher reward is worse

     local rewardDifference = reward:clone():add(-rewardBaseline, ones) 

     local baselineScores
     if USE_BIDIR_BASELINE then
        assert(neatQA.CONDITION ~= "mixed")
        bidir_baseline_gradparams:zero()
        local inputsWithActions = {}
        for i=1,  neatQA.maximalLengthOccurringInInput[1] do
            inputsWithActions[i] = joinTable[i]:forward({neatQA.inputTensors[i]:view(-1,1), attention_decisions[i]:view(-1,1)})
        end
        baselineScores = bidir_baseline:forward(inputsWithActions)
        for i=1, neatQA.maximalLengthOccurringInInput[1]  do
          local target
          if i == neatQA.maximalLengthOccurringInInput[1]  then
             target = reward:view(params.batch_size,1)
         else
             target = baselineScores[i+1]:view(params.batch_size,1)
          end

             if i == neatQA.maximalLengthOccurringInInput[1] and torch.uniform() > 0.93 then
               print("BIDIR "..i)
               print("40624 TARGET ")
               print(target)

               print("SCORES ")
               print(baselineScores[i])
             end
          bidirBaseline.criterion_gradients[i]:copy(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          if i == params.seq_length then
             --print(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          end
       end
       bidir_baseline:backward(inputsWithActions, auxiliary.shortenTable(bidirBaseline.criterion_gradients,  neatQA.maximalLengthOccurringInInput[1]))

       auxiliary.clipGradients(bidir_baseline_gradparams)
       bidir_baseline_params:add(-0.0008, bidir_baseline_gradparams) -- this is changed from the NEAT version, which first multiplied in place
     elseif USE_SIMPLE_BASELINE then
        assert(true or neatQA.CONDITION == "mixed")
        simple_baseline_gradparams:zero()
        local baselineInputs = neatQA.condition_mask

        baselineScores = simple_baseline:forward(baselineInputs)
          local target
             target = reward:view(params.batch_size,1)

             if torch.uniform() > 0.9 then
               print("SIMPLE BASELINE")
               print("40624 TARGET ")
               print(target)

               print("SCORES ")
               print(baselineScores)
               print("CONDITIONS ")
               print(neatQA.condition_mask)
             end
       simple_baseline:backward(baselineInputs, baseline_criterion:backward(baselineScores:view(-1,1), target))
       auxiliary.clipGradients(simple_baseline_gradparams)
       simple_baseline_params:add(-0.01, simple_baseline_gradparams) -- this is changed from the NEAT version, which first multiplied in place
     end

     -- this mean is not actually used, what is used is the bidirectional predictor
     rewardBaseline = 0.95 * rewardBaseline + 0.05 * (torch.sum(reward) * 1.0/params.batch_size)
     for i = neatQA.maximalLengthOccurringInInput[1], 1, -1 do
        rewardDifferenceForI = rewardDifference
        if USE_BIDIR_BASELINE then
            if i>1 then
              rewardDifferenceForI = baselineScores[i-1]:clone():add(-1,reward):mul(-1)
if true and torch.uniform() > 0.9 and i == 100 then
 print("4349  "..i.."   "..(torch.mean(torch.abs(rewardDifference) - torch.abs(rewardDifferenceForI))))
end
            end
        elseif USE_SIMPLE_BASELINE then
              rewardDifferenceForI = baselineScores:clone():add(-1,reward):mul(-1)
              if true and torch.uniform() > 0.9 and i == 100 then
                  print("2162  ".."   "..(torch.mean(torch.abs(rewardDifference) - torch.abs(rewardDifferenceForI))))
              end
        else
          assert(false)
        end

        local whatToMultiplyToTheFinalDerivative
        if (not neatQA.better_logsigmoid_gradients) then
           whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        end
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
-- it is important to take the probabilities and not the scores, since it will all be fed into the probabilities port of the attention moduke
           if (neatQA.better_logsigmoid_gradients) then
              -- note that entropy is positive, so there is a double - here and the term should enter positively
              attentionEntropyFactor[j] = params.ENTROPY_PENALTY * (math.exp(attentionObjects.probabilities[i][j][1]) *(attentionObjects.probabilities[i][j][1] - math.log(1 - math.exp(attentionObjects.probabilities[i][j][1]))))
--              print("..11824")
  --            print(attentionEntropyFactor[j])
    --          print(math.exp(attentionObjects.probabilities[i][j][1]))
           else
              --assert(false)
              attentionEntropyFactor[j] =  params.ENTROPY_PENALTY * (math.log(attentionObjects.probabilities[i][j][1]) - math.log(1 - attentionObjects.probabilities[i][j][1]))
           end
--           print("161")
  --         print(params.ENTROPY_PENALTY)
    --       print(attentionEntropyFactor[j])
      --     print(attentionObjects.probabilities[i][j][1])
        --   print(attentionObjects.scores[i][j][1])
          -- print(attentionObjects.decisions[i][j][1])
           if (not neatQA.better_logsigmoid_gradients) then
              --assert(false)

              whatToMultiplyToTheFinalDerivative[j] = 1 / (attentionObjects.probabilities[i][j][1]) -- THIS is an important part of REINFORCE!
           end
        end

        local factorsForTheDerivatives =  rewardDifferenceForI:clone() -- positive is worse?!
       if (not neatQA.better_logsigmoid_gradients) then
        --      assert(false)

             factorsForTheDerivatives:cmul(whatToMultiplyToTheFinalDerivative)
       end
        factorsForTheDerivatives:add(attentionEntropyFactor)
       local attentionNetwork = attentionObjects.attentionNetworks[i]
       local originalInputTensor = attentionObjects.originalInputTensors[i]
       local forwardHidden = attentionObjects.forwardHidden[i-1]
       local forwardCell = attentionObjects.forwardCell[i-1]
       local attentionArguments = {originalInputTensor}
       if true or neatQA.CONDITION == "preview" or neatQA.CONDITION == "mixed" then
        if neatQA.ATTENTION_DOES_Q_ATTENTION then
           table.insert(attentionArguments, neatQA.questionEmbeddings)
        elseif params.useBackwardQForAttention then
          assert(false)
          table.insert(attentionArguments,attentionObjects.questionBackward)
        else
          assert(false)
          table.insert(attentionArguments,attentionObjects.questionForward)
        end
        if neatQA.USE_INNOVATIVE_ATTENTION then
             assert(false)
             table.insert(attentionArguments, attentionObjects.questionBackward)
        end
       end
       table.insert(attentionArguments,forwardHidden)
       table.insert(attentionArguments,forwardCell)
       table.insert(attentionArguments,neatQA.questionHistory[i-1])
       table.insert(attentionArguments, neatQA.positionTensors[i])
       table.insert(attentionArguments, neatQA.questionHistoryFromSource[i-1])
       if true or neatQA.CONDITION == "mixed" then
          table.insert(attentionArguments, neatQA.condition_mask)
       end
       local firstGradient -- the attention score
       local secondGradient = torch.CudaTensor() -- the decision
       local thirdGradient = factorsForTheDerivatives -- the log probability
       local fourthGradient = torch.CudaTensor()
       if neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS then
         firstGradient = gradientWRTAttentionScore
       else
         firstGradient = torch.CudaTensor()
         firstGradient:zero()
       end
       local fifthGradient = questionHistoryGradient

       local sixthGradient = questionHistoryFromSourceGradient

       local gradientFactors = {firstGradient, secondGradient, thirdGradient, fourthGradient, fifthGradient, sixthGradient}
       if false then
         table.insert(gradientFactors, torch.CudaTensor())
         table.insert(gradientFactors, torch.CudaTensor())
       end
       local gradientAttention = attentionNetworks[i]:backward(attentionArguments,gradientFactors)
       gradientsAttention[i] = {questionForward = gradientAttention[2], forwardHidden = gradientAttention[3], forwardCell = gradientAttention[4]}
       questionHistoryGradient = gradientAttention[5]
       questionHistoryFromSourceGradient = gradientAttention[6]
     end
     assert(norm_dwRA == norm_dwRA)


if neatQA.APPLY_L2_REGULARIZATION_TO_ATT then
  ap, gap = attentionNetworks[1]:parameters()
  for i=1,#gap do
    if gap[i]:norm()>0 then -- exclude parameters that are not trained (e.g., embeddings)
      gap[i]:add(params.l2_regularization, ap[i])
    end
  end
end


if false then
      auxiliary.normalizeGradients(paramdxRA)
else
      auxiliary.clipGradients(paramdxRA)
end

if true then
      print("23615")
      print(paramdxRA:norm())
end


if true and torch.uniform() > 0.95 then
  ap, gap = attentionNetworks[1]:parameters()
  print("--24113")
  print(ap)
  print(gap)
  for i=1,#gap do
    print(i.."  "..gap[i]:norm().."  "..ap[i]:norm())
  end
end


if neatQA.USE_ADAM_FOR_REINFORCE then
    adam.adam_step = adam.adam_step + 1
    local beta1 = 0.9
    local beta2 = 0.999
    local epsilon = 0.00000001
    local alpha = params.lr_att
--    print("...")
--    print(paramdxRA[paramdxRA:storage():size()-1])
    adam.movingAverageOfGradients:mul(beta1):add(1-beta1,paramdxRA)
--    print(adam.movingAverageOfGradients[paramdxRA:storage():size()-1])

    paramdxRA:pow(2)
--    print(adam.square_gradient[paramdxRA:storage():size()-1])

    adam.movingAverageOfSquareGradients:mul(beta2):add(1-beta2,paramdxRA)
--    print(adam.movingAverageOfSquareGradients[paramdxRA:storage():size()-1])

    torch.sqrt(adam.temp, adam.movingAverageOfSquareGradients)

--    print(adam.temp[paramdxRA:storage():size()-1])

    adam.temp:add(epsilon)
    paramdxRA:zero()
    paramxRA:addcdiv(- math.sqrt(1-math.pow(beta2,adam.adam_step)) * alpha /(1-math.pow(beta1,adam.adam_step)), adam.movingAverageOfGradients, adam.temp)
--    print(paramxRA[paramdxRA:storage():size()-1])

else 
      auxiliary.updateParametersWithMomentum(paramxRA,paramdxRA,params.lr_att)
end

  end
end  

return gradientsAttention
  
end






