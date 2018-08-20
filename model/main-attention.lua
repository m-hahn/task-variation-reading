

BASE_DIRECTORY = "/u/scr/mhahn/"


BlockGradientLayer = require('nn.BlockGradientLayer')

require('globalForExpOutput')


--require('phono')
--require('linearization')
require('rnn')
require('soft-attention')
require('q-attention')
require('bidirBaseline')
require('simpleBaseline')
require('qa')


require('cunn')

require('nngraph')
require('base')
require('data') --ptb

-- SET THE PARAMETERS
require('setParameters')
--

require('readDict')



require('datasets')


------------------------

require 'lfs'


require('storeAnnotation')



assert(not ((not DOING_EVALUATION_OUTPUT) and (not DO_TRAINING)))

------------------------


-- READ THE DICTIONARY


require('readChunks')

require('numericalValues')

require('readFiles')

require('lstm')

require('attention')

require('auxiliary')

require('autoencoding')

require('combined')

--require('langmod')

require('neat-qa-SplitInput')

require('binning')

local function setup()
   if params.TASK == 'autoencoding' or params.TASK == 'combined' then
      return autoencoding.setupAutoencoding()
   elseif params.TASK == 'langmod' then
      return setupLanguageModelling()
   elseif params.TASK == 'qa' then
      return setupQA()
   elseif params.TASK == 'phono' then
      return setupPhono()
   elseif params.TASK == 'linearization' then
      return setupLinearization()
   elseif params.TASK == 'combined-soft' then
      return setupCombinedSoft()
   elseif params.TASK == 'combined-q' then
      return setupCombinedQ()
   elseif params.TASK == 'neat-qa' then
      return neatQA.setup()
   else
      print(TASK)
      crash()
   end
end



local function reset_state(state)

end

-- TODO make more efficient by filling zeros in place with :zero? but note that these things are from inside some neural network
function reset_ds()



   if params.TASK == 'autoencoding' then
    model.dsR[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsR[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h


    model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h
  elseif params.TASK == 'neat-qa' then
    model.dsR[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsR[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    if neatQA.ALSO_DO_LANGUAGE_MODELING then
      model.dsR[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h
    end

    --model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    --model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    --model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h


--   elseif params.TASK == "combined-q" then
-- no need to do anything
   elseif params.TASK == 'combined' or params.TASK == 'combined-soft' or params.TASK == 'combined-q' then
    model.dsR[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsR[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsR[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h


    model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h

   elseif params.TASK == 'langmod' then

    model.dsA[1] = torch.zeros(params.batch_size,params.rnn_size):cuda() --c
    model.dsA[2] = torch.zeros(params.batch_size,params.rnn_size):cuda() --h
    model.dsA[3] = torch.zeros(params.batch_size,params.vocab_size):cuda() --h

   elseif params.TASK == 'qa' then

  model.dsRT[1] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
  model.dsRT[2] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))

  model.dsRQ[1] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
  model.dsRQ[2] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))

  model.dsAt[1] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
  model.dsAt[2] = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
  model.dsAt[3] = transfer_data(torch.zeros(params.batch_size,params.rnn_size)) -- NOTE actually will later have different size

  model.dsAn[1] = transfer_data(torch.zeros(params.batch_size,NUMBER_OF_ANSWER_OPTIONS))
   else
      print(TASK)
      crash()
   end



end

--------------------------------------
--------------------------------------

local function fp(corpus, startIndex, endIndex)
   if params.TASK == 'autoencoding' or params.TASK == 'combined' then
      return autoencoding.fpAutoencoding(corpus, startIndex, endIndex)
   elseif params.TASK == 'langmod' then
      return fpLanguageModelling(corpus, startIndex, endIndex)
   elseif params.TASK == 'qa' then
      return fpQA(corpus, startIndex, endIndex)
   elseif params.TASK == 'phono' then
      return fpPhono(corpus, startIndex, endIndex)
   elseif params.TASK == 'linearization' then
      return fpLinearization(corpus, startIndex, endIndex)
   elseif params.TASK == 'combined-soft' then
      return fpCombinedSoft(corpus, startIndex, endIndex)
   elseif params.TASK == 'combined-q' then
      return combined.fpCombinedQ(corpus, startIndex, endIndex)
   elseif params.TASK == 'neat-qa' then
      return neatQA.fp(corpus, startIndex, endIndex)
   else
      print(params.TASK)
      crash()
   end
end


---------------------------------------
---------------------------------------



local function bp(corpus, startIndex, endIndex)
   if params.TASK == 'autoencoding' then
      return autoencoding.bpAutoencoding(corpus, startIndex, endIndex)
   elseif params.TASK == 'combined' then
      if USE_BASELINE_NETWORK then
         return combined.bpCombined(corpus, startIndex, endIndex)
      else
         return combined.bpCombinedNoBaselineNetwork(corpus, startIndex, endIndex)
      end
   elseif params.TASK == 'langmod' then
      return bpLanguageModelling(corpus, startIndex, endIndex)
   elseif params.TASK == 'qa' then
      return bpQA(corpus, startIndex, endIndex)
   elseif params.TASK == 'phono' then
      return bpPhono(corpus, startIndex, endIndex)
   elseif params.TASK == 'linearization' then
      return bpLinearization(corpus, startIndex, endIndex)
   elseif params.TASK == 'combined-soft' then
      return combined.bpCombinedSoft(corpus, startIndex, endIndex)
   elseif params.TASK == 'combined-q' then
      return combined.bpCombinedQ(corpus, startIndex, endIndex)
   elseif params.TASK == 'neat-qa' then
      return neatQA.bp(corpus, startIndex, endIndex)
   else
      print(TASK)
      crash()
   end
end

require('getParams')

require('nn.UniformLayer')

require('nn.ScalarMult')

require('nn.BlockGradientLayer')


local function tryReadParam(func)
if not pcall(func) then
print("ERROR ")
print(func)
end
end




local function main()




  g_init_gpu({params.gpu_number})

  if params.TASK == 'neat-qa' then
      if CREATE_RECORDED_FILES and DOING_EVALUATION_OUTPUT then
        PRINTING_PERIOD = 1
      else
        PRINTING_PERIOD = 21 -- 51 -- 51 --5 --21 --21
      end
  else
        PRINTING_PERIOD = 21 --51 --51
  end
  if false and DOING_DEBUGGING then
     PRINTING_PERIOD = 1
  end
  
  readDict.readDictionary()

  if params.TASK == 'neat-qa' then
     readDict.createNumbersToEntityIDsIndex()
  end


  print("Network parameters:")
  print(params)

  print("setup")
  setup()
  print("setup done")



  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local numberOfWords = 0
  local counter = 0


  tryReadParam(getLearningRateFromFile)
  tryReadParam(getAttentionFromFile)
  tryReadParam(getAttentionLearningRateFromFile)
  tryReadParam(getEntropyWeightFromFile)
  tryReadParam(getTotalAttentionWeightFromFile)
  tryReadParam(getBaselineFromFile)
  tryReadParam(getL2RegFromFile)



  print(FIXED_ATTENTION)
  print(params.lr)
  print(params.lr_att)



  for epoch = 1,EPOCHS_NUMBER do
     local IS_LAST_EPOCH
   
       IS_LAST_EPOCH = (epoch == EPOCHS_NUMBER)
    
--     print("--------"..epoch)
     if (not DO_TRAINING) and epoch > 1 then
       print("BREAK 301. Not doing training, so only doing the first epoch.")
       break
     end

     readChunks.resetFileIterator()

     epochCounter = epoch

--     print("31316")
     --[[print(files)
     print(#files)
     print(params.batch_size)
     print(#files-params.batch_size+1)]]
     --print(4756)
     --print(hasNextFile())
      --print("316  "..tostring(readChunks.hasNextFile()))


     while readChunks.hasNextFile() do
        local lastIteration = false  

        if ( PERCENTAGE_OF_DATA < (100.0 * (readChunks.corpusReading.currentFile+0.0) / #readChunks.files)) then
          print("AT 313")
          print( (100.0 * (readChunks.corpusReading.currentFile+0.0) / #readChunks.files))
          lastIteration = true
        end

       
     --print(i)
  
      for l = 1, params.batch_size do
         --corpus[l] = readAFile(files[i+l-1])
         if params.TASK == 'langmod' or params.TASK == 'autoencoding' or params.TASK == 'combined' or params.TASK == 'combined-soft' or params.TASK == "combined-q" then
               --print(l)
               --print("339")

             if binning.do_binning then
               assert(params.TASK == "neat-qa", "Do you really want to call binning when doing autoencoding? I'm not sure whether this thing here is intended or not.")
               readChunks.corpus[l] = binning.getNext(readChunks.readNextChunkForBatchItem,l,     (function(x) return (#(x)) end)   )
             else
               readChunks.corpus[l] = readChunks.readNextChunkForBatchItem(l)
             end
             if params.INCLUDE_NUMERICAL_VALUES then -- huhu
                if binning.do_binning then
                  print("THIS PROBABLY DOES NOT WORK CORRECTLY. 344")
                  assert(false)
                end
                  numericalValues.getNumericalValuesForBatchItem(l)
             end
         elseif params.TASK == 'neat-qa' then
           if binning.do_binning then
             readChunks.corpus[l] = binning.getNext(readFiles.readNextQAItem,l)
           else
             readChunks.corpus[l] = readFiles.readNextQAItem(l)
           end
         else
             crash()
         end
      end



--      print("BEFORE FORWARD")
      --print(readChunks.corpus)
      local perp, actor_output = fp(readChunks.corpus, 1, params.batch_size)
  --    print("DONE FORWARD")
      if params.TASK == 'neat-qa' then
         numberOfWords = numberOfWords + params.batch_size * neatQA.maximalLengthOccurringInInput[1]
      else
         numberOfWords = numberOfWords + params.batch_size * params.seq_length
      end



      if MAKE_SKIPPING_STATISTICS then
           updateSkippingStatistics(readChunks.corpus)
      end

      if STORE_ATTENTION_ANNOTATION then
         if params.TASK == "neat-qa" then
          storeAttentionAnnotationQA(readChunks.corpus)
         else
          storeAttentionAnnotation(readChunks.corpus)
         end
      end

      if WRITE_SURPRISAL_SCORES then
          storeSurprisalScores(readChunks.corpus)
      end


      if DOING_EVALUATION_OUTPUT then
          for  l=1, params.batch_size do
              if readChunks.corpus[l][1] == 2 and readChunks.corpus[l][2] == 2 and readChunks.corpus[l][3] == 2 then
                 print(readChunks.corpus[l])
              else
                 evaluationAccumulators.reconstruction_loglikelihood = evaluationAccumulators.reconstruction_loglikelihood + perp[l]
                 if params.TASK == "combined" or params.TASK == "combined-soft"or params.TASK == "combined-q" then
                     evaluationAccumulators.lm_loglikelihood = evaluationAccumulators.lm_loglikelihood + nll_reader[l]
                 end
--                 print("3342   "..files[readChunks.corpusReading.locationsOfLastBatchChunks[l].file])
                 evaluationAccumulators.numberOfTokens = evaluationAccumulators.numberOfTokens + params.seq_length
              end
          end
      end

      if params.TASK == 'autoencoding' or params.TASK == 'neat-qa' then
           meanTotalAtt = 0.8 * meanTotalAtt + 0.2 * torch.sum(totalAttentions) * 1/params.batch_size
      end

      if params.TASK == 'autoencoding' or params.TASK == 'langmod' or params.TASK == 'qa' then
           meanNLL = 0.8 * meanNLL + 0.2 * torch.sum(perp) * 1/params.batch_size
      end

      --print(meanNLL)
      counter = counter + 1

      if  counter % 100 == 0 and TASK == 'qa' then
         qaCorrect = 0
         qaIncorrect = 0
      end
      -- print stats
      fileStats:write((counter*params.batch_size)..'\t'..perp:mean()..'\n')
      fileStats:flush()


      if counter % PRINTING_PERIOD == 0 then
         print('WORDS '..numberOfWords..'  EPOCH '..epoch..'  '..(100.0 * (readChunks.corpusReading.currentFile+0.0) / #readChunks.files))
         print("WORDS/SEC   "..numberOfWords / torch.toc(start_time))
         local since_beginning = g_d(torch.toc(beginning_time) / 60)


         if params.TASK == 'autoencoding' then
assert(false)
            print("+++++++ "..perp[1]..'  '..meanNLL..'  '..meanTotalAtt..'  '..(params.TOTAL_ATTENTIONS_WEIGHT * meanTotalAtt + meanNLL))
             print(epoch.."  "..corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(probabilityOfChoices[1]..'  '..totalAttentions[1])
            print(experimentNameOut)
            print(params) 
            if probabilityOfChoices[1] ~= probabilityOfChoices[1]  then
               crash()
            end
   
            if MAKE_SKIPPING_STATISTICS then
                printSkippingStatistics()
            end
   
   
            for l = 1, 1 do
               print("....")
               print(perp[l])
               for j=1,params.seq_length do
                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
                  io.write((readDict.chars[getFromData(readChunks.corpus,l,j)]))--..'\n')
                  io.write(" ~ "..readDict.chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(readChunks.corpus,l,j)]).."  "..attention_decisions[j][l].."  "..attention_scores[j][l][1])
                  if params.INCLUDE_SURPRISAL_VALUES then
                     io.write("  "..numericalValues.numericalValuesImporter.values[1][j])
                  end
                  io.write("\n")
               end
            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp:mean()..'\n')
            fileStats:flush()
            --io.output(stdout)
         elseif params.TASK == 'combined' then
            combined.printStuffForCombined(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'combined-soft' then
            printStuffForCombinedSoft(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'combined-q' then
            printStuffForCombinedQ(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'langmod' then
            printStuffForLangmod(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'qa' then
            printStuffForQA(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'phono' then
            printStuffForPhono(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'linearization' then
           printStuffForLinearization(perp, actor_output, since_beginning, epoch, numberOfWords)
         elseif params.TASK == 'neat-qa' then
           neatQA.printStuff(perp, actor_output, since_beginning, epoch, numberOfWords)
         else
            print(params.TASK)
            crash()
         end


--         if not use_attention_network then
  --          if not pcall(getAttentionFromFile) then
    --           print("ERROR IN FILE ATT")
      --      end
        -- end

  tryReadParam(getLearningRateFromFile)
  tryReadParam(getAttentionFromFile)
  tryReadParam(getAttentionLearningRateFromFile)
  tryReadParam(getEntropyWeightFromFile)
  tryReadParam(getTotalAttentionWeightFromFile)
  tryReadParam(getBaselineFromFile)
  tryReadParam(getL2RegFromFile)

         print(FIXED_ATTENTION)
         print(params.lr)
         print(params.lr_att)
         print(params.lr_att)
         print(params.TOTAL_ATTENTIONS_WEIGHT)
         print(params.l2_regularization)

      fileStatsLrAtt:write((counter*params.batch_size)..'\t'..params.lr_att..'\n')
      fileStatsLrAtt:flush()

      fileStatsEntropyW:write((counter*params.batch_size)..'\t'..params.ENTROPY_PENALTY..'\n')
      fileStatsEntropyW:flush()

      fileStatsTotAttW:write((counter*params.batch_size)..'\t'..params.TOTAL_ATTENTIONS_WEIGHT..'\n')
      fileStatsTotAttW:flush()





      end

neatQA.sessionCounter = counter

      if DO_TRAINING then
         bp(readChunks.corpus, 1, params.batch_size)
      else
        local attentionObjects = {attentionNetworks = attentionNetworks,questionForward = question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], questionBackward=question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], decisions=attention_decisions,scores=attention_scores, originalInputTensors = neatQA.inputTensors, probabilities = attention_probabilities,questionInputTensors=neatQA.inputTensorsQuestion}
 neatQA.numberOfExamples = neatQA.numberOfExamples + 1


     local logLikelihoodLoss = transfer_data(nll)

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






-- OLD VERSION
if false then
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
end

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
        print("Tot. Number: "..(params.batch_size*neatQA.numberOfExamples))



      end

if false then

-- gradient check
local epsilon = 0.01
local parameters, gradParameters = neatQA.parameters()
local perp0, actor_output = fp(readChunks.corpus, 1, params.batch_size)

for b=1,#parameters do
 for c=1,#(parameters[b]) do
  local storage = parameters[b][c]:storage()
  for d=1,storage:size() do
   print("52410 "..tostring(d))
   storage[d] = storage[d]-epsilon
   local perpMinus ,_ = fp(readChunks.corpus,1,params.batch_size):sum()
   storage[d] = storage[d]+2*epsilon
   local perpPlus ,_ = fp(readChunks.corpus,1,params.batch_size):sum()
   local perpPlus2 ,_ = fp(readChunks.corpus,1,params.batch_size):sum()
   print("~~")
   print(perpPlus)
   print(perpPlus2)
   storage[d] = storage[d]-epsilon
   numericalGradient = (perpPlus - perpMinus)/(2*epsilon)
   print("..")
   print(perpPlus)
   print(perpMinus)
--   print(gradParameters[b][c])
   local computedGradient = gradParameters[b][c]:storage()[d]
   print(tostring(b).."  "..tostring(c).."  "..tostring(d).."  "..tostring(numericalGradient).."  "..tostring(computedGradient))

  end
 end
end

end






      if counter % 33 == 0 then
        cutorch.synchronize()
        collectgarbage()
      end


      if DO_TRAINING and ((lastIteration and IS_LAST_EPOCH) or (counter % PRINT_MODEL_PERIOD == 0)) then
        print("WRITING MODEL...")
        local modelsArray
        if params.TASK == 'neat-qa' then
           local SparamxForward, SparamdxForward = forward_network:parameters()
           local SparamxBackward, SparamdxBackward = backward_network:parameters()
           local SparamxQForward, SparamdxQForward = question_forward_network:parameters()
           local SparamxQBackward, SparamdxQBackward = question_backward_network:parameters()


           local SparamxA, SparamdxA = actor_core_network:parameters()
           local SparamxRA, SparamdxRA = table.unpack({{},{}})
           if neatQA.USE_ATTENTION_NETWORK then
              SparamxRA, SparamdxRA= attentionNetwork:parameters()
           end

           modelsArray = {params = params,readWords = (numberOfWords/params.seq_length), SparamxForward = SparamxForward, SparamdxForward = SparamdxForward, SparamxBackward = SparamxBackward, SparamdxBackward = SparamdxBackward, SparamxQForward = SparamxQForward, SparamdxQForward = SparamdxQForward, SparamxQBackward = SparamxQBackward, SparamdxQBackward = SparamdxQBackward, SparamxA = SparamxA, SparamdxA = SparamdxA, SparamxRA = SparamxRA, SparamdxRA = SparamdxRA}

          else 
            assert(false)
          end

           assert(not(USE_BIDIR_BASELINE and USE_SIMPLE_BASELINE))
           if USE_BIDIR_BASELINE and bidir_baseline ~= nil then
             local uB, udB = bidir_baseline:parameters()
             table.insert(modelsArray, uB)
             table.insert(modelsArray, udB)
           end
           if USE_SIMPLE_BASELINE and simple_baseline ~= nil then
             local uB, udB = simple_baseline:parameters()
             table.insert(modelsArray, uB)
             table.insert(modelsArray, udB)
           end
         if modelsArray ~= nil then
            assert(OVERWRITE_MODEL)
            torch.save(BASE_DIRECTORY..'/model-'..experimentNameOut, modelsArray, "binary")
         end
      end

      if DOING_EVALUATION_OUTPUT then
            print(evaluationAccumulators.reconstruction_loglikelihood.."&\n"..evaluationAccumulators.lm_loglikelihood.."\n") 
      end

      if lastIteration then
         print("Break following 313")
         break
      end

    end
  end
  print("Training is over.")


    if DOING_EVALUATION_OUTPUT then
             PERP_ANNOTATION_FILE = DATASET_DIR.."/annotation/perp-"..experimentNameOut..".txt"

             local fileOut = io.open(PERP_ANNOTATION_FILE, "w")
             print(PERP_ANNOTATION_FILE)
             tokenCountForLM = 49/50 * evaluationAccumulators.numberOfTokens
             fileOut:write("REC "..(evaluationAccumulators.reconstruction_loglikelihood) .."\n".."LM  "..(evaluationAccumulators.lm_loglikelihood).."\n".."REC "..(evaluationAccumulators.reconstruction_loglikelihood/evaluationAccumulators.numberOfTokens) .."\n".."LM  "..(evaluationAccumulators.lm_loglikelihood/tokenCountForLM) .."\n".."REC "..math.exp(evaluationAccumulators.reconstruction_loglikelihood/evaluationAccumulators.numberOfTokens) .."\n".."LM  "..math.exp(evaluationAccumulators.lm_loglikelihood/tokenCountForLM).."\n") 
             fileOut:close()
      end




  fileStats:close()



end

if (not OVERWRITE_MODEL) and  (lfs.attributes(BASE_DIRECTORY..'/model-'..experimentNameOut) ~= nil) then
   print("MODEL EXISTS, ABORTING")
else
  if (lfs.attributes(BASE_DIRECTORY..'/model-'..experimentNameOut) ~= nil) and OVERWRITE_MODEL then
     print("OVERWRITE MODEL")
  end
   main()
end
