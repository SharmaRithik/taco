#include "lower.h"

#include <vector>

#include "lower_scalar_expression.h"
#include "lower_util.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_rule.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "component_types.h"
#include "ir.h"
#include "ir_visitor.h"
#include "ir_codegen.h"
#include "var.h"
#include "storage/iterator.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

using namespace taco::ir;

using taco::internal::Tensor;
using taco::ir::Expr;
using taco::ir::Var;
using taco::ir::Add;

struct Context {
  Context(const set<Property>&     properties,
          const IterationSchedule& schedule,
          const Iterators&         iterators,
          const map<Tensor,Expr>&  tensorVars,
          const size_t             allocSize)
      : properties(properties), schedule(schedule), iterators(iterators),
        tensorVars(tensorVars), allocSize(allocSize) {
  }

  const set<Property>&     properties;
  const IterationSchedule& schedule;
  const Iterators&         iterators;
  const map<Tensor,Expr>&  tensorVars;
  const size_t             allocSize;
};

static vector<Stmt> lower(const Expr& expr,
                          taco::Var indexVar,
                          vector<Expr> indexVars,
                          const Context& ctx) {
  MergeRule mergeRule = ctx.schedule.getMergeRule(indexVar);
  MergeLattice mergeLattice = MergeLattice::make(mergeRule);
  vector<TensorPathStep> mergeRuleSteps = mergeRule.getSteps();

  TensorPathStep resultStep = mergeRule.getResultStep();
  storage::Iterator resultIterator = (resultStep.getPath().defined())
                                     ? ctx.iterators.getIterator(resultStep)
                                     : storage::Iterator();

  Tensor resultTensor = ctx.schedule.getTensor();
  Expr resultTensorVar = ctx.tensorVars.at(resultTensor);

  // Turn of merging if there's one or zero sparse arguments
  int sparseOperands = 0;
  for (auto& step : mergeRuleSteps) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() == LevelType::Sparse) {
      sparseOperands++;
    }
  }
  bool merge = (sparseOperands > 1);

  bool reduceToVar = (indexVar.isReduction() &&
                      !ctx.schedule.hasFreeVariableDescendant(indexVar));


  // Begin code generation
  vector<Stmt> code;

  code.push_back(BlankLine::make());
  code.push_back(Comment::make(" --------------------------------- " +
                               toString(indexVar) +
                               " ---------------------------------"));

  // Emit code to initialize ptr variables
  if (merge) {
    for (auto& step : mergeRuleSteps) {
      storage::Iterator iterator = ctx.iterators.getIterator(step);
      Expr ptr = iterator.getPtrVar();

      storage::Iterator iteratorPrev = ctx.iterators.getPreviousIterator(step);
      Expr ptrPrev = iteratorPrev.getPtrVar();

      Tensor tensor = step.getPath().getTensor();
      Expr tvar = ctx.tensorVars.at(tensor);

      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin());

      code.push_back(iteratorInit);
    }
  }

  // Emit code to initialize reduction variable (if reduction loop)
  Expr reductionVar;
  if (reduceToVar) {
    string name = "t" + indexVar.getName();
    reductionVar = Var::make(name, typeOf<double>(), false);
    Stmt reductionVarInit = VarAssign::make(reductionVar, 0.0);
    util::append(code, {reductionVarInit});
  }

  // Emit one loop per lattice point lp
  vector<Stmt> mergeLoops;
  auto latticePoints = mergeLattice.getPoints();
  for (MergeLatticePoint lp : latticePoints) {
    vector<TensorPathStep> lpSteps = lp.getSteps();
    vector<Stmt> loopBody;

    // Emit code to initialize sparse idx variables
    map<TensorPathStep, Expr> tensorIdxVariables;
    vector<Expr> tensorIdxVariablesVector;
    for (TensorPathStep& step : lpSteps) {
      Format format = step.getPath().getTensor().getFormat();
      if (format.getLevels()[step.getStep()].getType() == LevelType::Sparse) {
        storage::Iterator iterator = ctx.iterators.getIterator(step);

        Expr idxStep = iterator.getIdxVar();
        tensorIdxVariables.insert({step, idxStep});
        tensorIdxVariablesVector.push_back(idxStep);

        Stmt initDerivedVars = iterator.initDerivedVar();
        loopBody.push_back(initDerivedVars);
      }
    }

    if (tensorIdxVariablesVector.size() == 0) {
      iassert(lpSteps.size() > 0);
      Expr idxStep = ctx.iterators.getIterator(lpSteps[0]).getIdxVar();
      tensorIdxVariablesVector.push_back(idxStep);
    }

    // Emit code to initialize the index variable (min of path index variables)
    Expr idx ;
    if (merge) {
      idx = Var::make(indexVar.getName(), typeOf<int>(), false);
      Stmt initIdxStmt = mergePathIndexVars(idx, tensorIdxVariablesVector);
      loopBody.push_back(initIdxStmt);
    }
    else {
      idx = tensorIdxVariablesVector[0];
      const_cast<Var*>(idx.as<Var>())->name = indexVar.getName();
    }

    // Emit code to initialize dense ptr variables
    for (TensorPathStep& step : lpSteps) {
      Format format = step.getPath().getTensor().getFormat();
      if (format.getLevels()[step.getStep()].getType() == LevelType::Dense) {
        storage::Iterator iterator = ctx.iterators.getIterator(step);
        storage::Iterator iteratorPrev= ctx.iterators.getPreviousIterator(step);

        Expr stepIdx = iterator.getIdxVar();
        tensorIdxVariables.insert({step, stepIdx});

        Expr ptrVal = ir::Add::make(ir::Mul::make(iteratorPrev.getPtrVar(),
                                                  iterator.end()), idx);
        Stmt initPtr = VarAssign::make(iterator.getPtrVar(), ptrVal);
        loopBody.push_back(initPtr);
      }
    }
    if (resultIterator.defined() && resultIterator.isRandomAccess()) {
      auto resultPrevIterator = ctx.iterators.getPreviousIterator(resultStep);
      Expr ptrVal = ir::Add::make(ir::Mul::make(resultPrevIterator.getPtrVar(),
                                                resultIterator.end()), idx);
      Stmt initResultPtr = VarAssign::make(resultIterator.getPtrVar(), ptrVal);
      loopBody.push_back(initResultPtr);
    }
    loopBody.push_back(BlankLine::make());

    // Emit one case per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = mergeLattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : dominatedPoints) {
      vector<TensorPathStep> lqSteps = lq.getSteps();

      // Case expression
      vector<Expr> stepIdxEqIdx;
      vector<TensorPathStep> caseSteps = lq.simplify().getSteps();
      for (auto& caseStep : caseSteps) {
        stepIdxEqIdx.push_back(Eq::make(tensorIdxVariables.at(caseStep), idx));
      }
      Expr caseExpr = conjunction(stepIdxEqIdx);

      // Case body
      vector<Stmt> caseBody;
      indexVars.push_back(idx);

      // Print coordinate (only in base case)
      if (util::contains(ctx.properties, Print) &&
          ctx.schedule.getChildren(indexVar).size() == 0) {
        auto print = printCoordinate(indexVars);
        util::append(caseBody, print);
      }

      // Emit compute code. There are three cases:
      // Case 1: We still have free variables left to emit. We first emit
      //         code to compute available expressions and store them in
      //         temporaries, before we recurse on the next index variable.
      // Case 2: We are emitting the last free variable. We first recurse to
      //         compute remaining reduction variables into a temporary,
      //         before we compute and store the main expression
      // Case 3: We have emitted all free variables, and are emitting a
      //         summation variable. We first first recurse to emit remaining
      //         summation variables, before we add in the available
      //         expressions for the current summation variable.

      // Emit available sub-expressions at this level (case 1)
      if (util::contains(ctx.properties, Compute)) {
        if (ctx.schedule.hasFreeVariableDescendant(indexVar)) {
          caseBody.push_back(Comment::make("Emit available sub-expressions"));
        }
      }

      // Recursive call to emit iteration schedule children
      for (auto& child : ctx.schedule.getChildren(indexVar)) {
        auto childCode = lower(expr, child, indexVars, ctx);
        util::append(caseBody, childCode);
      }

      // Compute and store available expression (case 2,3)
      if (util::contains(ctx.properties, Compute)) {
        caseBody.push_back(Comment::make("Combine and store computed expressions"));
        set<TensorPathStep> stepsInLq(lqSteps.begin(), lqSteps.end());
        vector<TensorPathStep> stepsNotInLq;
        for (auto& step : mergeRuleSteps) {
          if (!util::contains(stepsInLq, step)) {
            stepsNotInLq.push_back(step);
          }
        }
        Expr computeExpr = removeExpressions(expr, stepsNotInLq, ctx.iterators);

        // Store result to result tensor (case 2) or reduction variable (case 3)
        if (ctx.schedule.isLastFreeVariable(indexVar)) {
          caseBody.push_back(Comment::make("a.vals[a1_ptr] = tk;"));

          auto resultPath = ctx.schedule.getResultTensorPath();
          storage::Iterator resultIterator = (resultTensor.getOrder() > 0)
              ? ctx.iterators.getIterator(resultPath.getLastStep())
              : ctx.iterators.getRootIterator();
          Expr resultPtr = resultIterator.getPtrVar();
          Expr vals = GetProperty::make(resultTensorVar,TensorProperty::Values);

          // If the last free variable has a reduction variable ancestor in the
          // iteration schedule tree then we must emit a compound store,
          // otherwise we emit a normal store
          Stmt compute = ctx.schedule.hasReductionVariableAncestor(indexVar)
              ? compoundStore(vals, resultPtr, computeExpr)
              : Store::make(vals, resultPtr, computeExpr);

          util::append(caseBody, {Comment::make(util::toString(compute))});
        }
        else if (reduceToVar) {
          caseBody.push_back(compoundAssign(reductionVar, computeExpr));
        }
      }

      // Emit code to compute result values in base case (DEPRECATED)
      if (util::contains(ctx.properties, Compute)) {
        if (ctx.schedule.getChildren(indexVar).size() == 0) {

          auto resultPath = ctx.schedule.getResultTensorPath();
          storage::Iterator resultIterator = (resultTensor.getOrder() > 0)
              ? ctx.iterators.getIterator(resultPath.getLastStep())
              : ctx.iterators.getRootIterator();
          Expr resultPtr = resultIterator.getPtrVar();

          // Build the index expression for this case
          set<TensorPathStep> stepsInLq(lqSteps.begin(), lqSteps.end());
          vector<TensorPathStep> stepsNotInLq;
          for (auto& step : mergeRuleSteps) {
            if (!util::contains(stepsInLq, step)) {
              stepsNotInLq.push_back(step);
            }
          }
          Expr subexpr = removeExpressions(expr, stepsNotInLq, ctx.iterators);
          Expr vals = GetProperty::make(resultTensorVar,TensorProperty::Values);
          Stmt compute = compoundStore(vals, resultPtr, subexpr);
          util::append(caseBody, {BlankLine::make(), compute});
        }
      }

      // Emit code to store the index variable value to idx
      if (util::contains(ctx.properties, Assemble) && resultIterator.defined()){
        Stmt idxStore = resultIterator.storeIdx(idx);
        if (idxStore.defined()) {
          if (util::contains(ctx.properties, Comment)) {
            Stmt comment = Comment::make("insert index value");
            util::append(caseBody, {BlankLine::make(), comment});
          }
          util::append(caseBody, {idxStore});
        }
      }

      // Emit code to increment the results iterator variable
      if (resultIterator.defined() && !resultIterator.isRandomAccess()) {
        Expr resultPtr = resultIterator.getPtrVar();
        Stmt ptrInc = VarAssign::make(resultPtr, Add::make(resultPtr, 1));

        Expr doResize = ir::And::make(
            Eq::make(0, BitAnd::make(Add::make(resultPtr, 1), resultPtr)),
            Lte::make(ctx.allocSize, Add::make(resultPtr, 1)));
        Expr newSize = ir::Mul::make(2, ir::Add::make(resultPtr, 1));
        Stmt resizeIndices = resultIterator.resizeIdxStorage(newSize);

        if (resultStep != resultStep.getPath().getLastStep()) {
          util::append(caseBody, {BlankLine::make()});
          storage::Iterator nextIterator =
              ctx.iterators.getNextIterator(resultStep);

          // Emit code to resize idx and ptr
          if (util::contains(ctx.properties, Assemble)) {
            Stmt resizePtr = nextIterator.resizePtrStorage(newSize);
            resizeIndices = Block::make({resizePtr, resizeIndices});
            resizeIndices = IfThenElse::make(doResize, resizeIndices);
            ptrInc = Block::make({ptrInc, resizeIndices});
          }

          Expr ptrArr = GetProperty::make(resultTensorVar,
                                          TensorProperty::Pointer,
                                          resultStep.getStep()+1);
          Expr producedVals =
              Gt::make(Load::make(ptrArr, Add::make(resultPtr,1)),
                       Load::make(ptrArr, resultPtr));
          ptrInc = IfThenElse::make(producedVals, ptrInc);
        } else if (util::contains(ctx.properties, Assemble)) {
          // Emit code to resize idx
          resizeIndices = IfThenElse::make(doResize, resizeIndices);
          ptrInc = Block::make({ptrInc, resizeIndices});
        }

        util::append(caseBody, {ptrInc});
      }

      indexVars.pop_back();
      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    
    Stmt caseStmt = !merge ? cases[0].second : Case::make(cases);
    loopBody.push_back(caseStmt);
    loopBody.push_back(BlankLine::make());

    // Emit code to conditionally increment iterator variables
    if (merge) {
      for (auto& step : lpSteps) {
        storage::Iterator iterator = ctx.iterators.getIterator(step);

        Expr iteratorVar = iterator.getIteratorVar();
        Expr tensorIdx = tensorIdxVariables.at(step);

        Stmt inc = VarAssign::make(iteratorVar, Add::make(iteratorVar, 1));
        Stmt maybeInc = IfThenElse::make(Eq::make(tensorIdx, idx), inc);
        loopBody.push_back(maybeInc);
      }
    }

    // Emit loop (while loop for merges and for loop for non-merges)
    Stmt loop;
    if (merge) {
      // Loop until any index has been exchaused
      vector<Expr> stepIterLqEnd;
      vector<TensorPathStep> mergeSteps = lp.simplify().getSteps();
      for (auto& mergeStep : mergeSteps) {
        storage::Iterator iter = ctx.iterators.getIterator(mergeStep);
        stepIterLqEnd.push_back(Lt::make(iter.getIteratorVar(), iter.end()));
      }
      Expr untilAnyExhausted = conjunction(stepIterLqEnd);
      loop = While::make(untilAnyExhausted, Block::make(loopBody));
    }
    else {
      iassert(lp.simplify().getSteps().size() == 1);
      storage::Iterator iter = ctx.iterators.getIterator(lpSteps[0]);
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody));
    }
    mergeLoops.push_back(loop);
    mergeLoops.push_back(BlankLine::make());
  }
  if (!merge) {
    iassert(mergeLoops.size() > 0);
    mergeLoops = {mergeLoops[0]};
  }
  util::append(code, mergeLoops);

  // Emit code to store the segment size to ptr
  if (util::contains(ctx.properties, Assemble) && resultIterator.defined()) {
    Stmt ptrStore = resultIterator.storePtr();
    if (ptrStore.defined()) {
      util::append(code, {ptrStore});
    }
  }

  code.push_back(Comment::make(" -------------------------------- /" +
                               toString(indexVar) +
                               " ---------------------------------"));
  code.push_back(BlankLine::make());
  return code;
}

Stmt lower(const Tensor& tensor, string funcName,
           const set<Property>& properties) {
  auto name = tensor.getName();
  auto vars = tensor.getIndexVars();
  auto indexExpr = tensor.getExpr();
  auto allocSize = tensor.getAllocSize();

  string exprString = name + "(" + util::join(vars) + ")" +
                      " = " + util::toString(indexExpr);

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVars;

  // Pack result tensor into output parameter list
  Expr tensorVar = Var::make(name, typeOf<double>(), tensor.getFormat());
  tensorVars.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<Tensor> operands = internal::getOperands(indexExpr);
  for (auto& operand : operands) {
    iassert(!util::contains(tensorVars, operand));
    Expr operandVar = Var::make(operand.getName(), typeOf<double>(),
                                operand.getFormat());
    tensorVars.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

  // Create the schedule and the iterators of the lowered code
  IterationSchedule schedule = IterationSchedule::make(tensor);
  Iterators iterators(schedule, tensorVars);

  // Initialize the result ptr variables
  vector<Stmt> resultPtrInit;
  for (auto& indexVar : tensor.getIndexVars()) {
    MergeRule mergeRule = schedule.getMergeRule(indexVar);

    TensorPathStep step = mergeRule.getResultStep();
    Tensor result = schedule.getResultTensorPath().getTensor();

    Expr tensorVar = tensorVars.at(result);
    storage::Iterator iterator = iterators.getIterator(step);
    Expr ptr = iterator.getPtrVar();
    Expr ptrPrev = iterators.getPreviousIterator(step).getPtrVar();

    // Emit code to initialize the result ptr variable
    Stmt iteratorInit = VarAssign::make(iterator.getPtrVar(), iterator.begin());
    resultPtrInit.push_back(iteratorInit);
  }

  // Lower the scalar expression
  Expr expr = lowerScalarExpression(indexExpr, iterators, schedule, tensorVars);

  // Lower the iteration schedule
  vector<Stmt> code;
  auto& roots = schedule.getRoots();

  // Lower scalar expressions
  if (roots.size() == 0 && util::contains(properties, Compute)) {
    Expr resultTensorVar = tensorVars.at(schedule.getTensor());
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    Stmt compute = Store::make(vals, 0, expr);
    code.push_back(compute);
  }
  // Lower tensor expressions
  else {
    Context ctx(properties, schedule, iterators, tensorVars, allocSize);
    for (auto& root : roots) {
      vector<Stmt> loopNest = lower(expr, root, {}, ctx);
      util::append(code, loopNest);
    }
  }

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), code.begin(), code.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
