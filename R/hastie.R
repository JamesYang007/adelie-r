reweight <- function(fam,weights){
    famname <-fam$name
    a <- attributes(fam)
    n <- length(a[["_weights"]])
if(length(weights)!=n)stop("replacement weight vector has to have same length as weight vector on object")
    weights <- weights/sum(weights)
    switch(famname,
           "gaussian" = glm.gaussian(
               y = a[["_y"]],
               weights = weights,
               opt = a[["_opt"]]
           ),
           "binomial_logit" = glm.binomial(
               y = a[["_y"]],
               weights = weights,
               link = "logit"
           ),
           "binomial_probit" = glm.binomial(
               y = a[["_y"]],
               weights = weights,
               link = "probit"
           ),
           "cox" = glm.cox(
               start = a[["_start"]],
               stop = a[["_stop"]],
               status = a[["_status"]],
               weights = weights,
               tie_method = a[["_tie_method"]]
           ),
           "multigaussian"= glm.multigaussian(
               y = a[["_y"]],
               weights = weights,
               opt = a[["_opt"]]
           ),
           "multinomial" = glm.multinomial(
               y = a[["_y"]],
               weights = weights
           ),
           "poisson" = glm.poisson(
               y = a[["_y"]],
               weights = weights
           )
           )
}
print.grpnet <- function (x, digits = max(3, getOption("digits") - 3), ...)
{
    cat("\nCall: ", deparse(x$call), "\n\n")
    state = x$state
    Df = rowSums(state$betas!=0)
    dev.ratio = state$devs
    lambdas=state$lmdas
    out = data.frame(Df, `%Dev` = round(dev.ratio *
        100, 2), Lambda = signif(lambdas, digits), check.names = FALSE,
        row.names = seq(along = Df))
    class(out) = c("anova", class(out))
    print(out)
}


#' make predictions from a "glmnet" object.
#'
#' Similar to other predict methods, this functions predicts fitted values,
#' logits, coefficients and more from a fitted \code{"glmnet"} object.
#'
#' The shape of the objects returned are different for \code{"multinomial"}
#' objects. This function actually calls \code{NextMethod()}, and the
#' appropriate predict method is invoked for each of the three model types.
#' \code{coef(...)} is equivalent to \code{predict(type="coefficients",...)}
#'
#' @aliases coef.glmnet coef.relaxed predict.glmnet predict.relaxed
#' predict.elnet predict.lognet predict.multnet predict.mrelnet predict.fishnet
#' predict.coxnet
#' @param object Fitted \code{"glmnet"} model object or a \code{"relaxed"}
#' model (which inherits from class "glmnet").
#' @param newx Matrix of new values for \code{x} at which predictions are to be
#' made. Must be a matrix; can be sparse as in \code{Matrix} package. This
#' argument is not used for \code{type=c("coefficients","nonzero")}
#' @param s Value(s) of the penalty parameter \code{lambda} at which
#' predictions are required. Default is the entire sequence used to create the
#' model.
#' @param type Type of prediction required. Type \code{"link"} gives the linear
#' predictors for \code{"binomial"}, \code{"multinomial"}, \code{"poisson"} or
#' \code{"cox"} models; for \code{"gaussian"} models it gives the fitted
#' values. Type \code{"response"} gives the fitted probabilities for
#' \code{"binomial"} or \code{"multinomial"}, fitted mean for \code{"poisson"}
#' and the fitted relative-risk for \code{"cox"}; for \code{"gaussian"} type
#' \code{"response"} is equivalent to type \code{"link"}. Type
#' \code{"coefficients"} computes the coefficients at the requested values for
#' \code{s}.  Note that for \code{"binomial"} models, results are returned only
#' for the class corresponding to the second level of the factor response.
#' Type \code{"class"} applies only to \code{"binomial"} or
#' \code{"multinomial"} models, and produces the class label corresponding to
#' the maximum probability. Type \code{"nonzero"} returns a list of the indices
#' of the nonzero coefficients for each value of \code{s}.
#' @param exact This argument is relevant only when predictions are made at
#' values of \code{s} (lambda) \emph{different} from those used in the fitting
#' of the original model. Not available for \code{"relaxed"} objects. If
#' \code{exact=FALSE} (default), then the predict function uses linear
#' interpolation to make predictions for values of \code{s} (lambda) that do
#' not coincide with those used in the fitting algorithm. While this is often a
#' good approximation, it can sometimes be a bit coarse.  With
#' \code{exact=TRUE}, these different values of \code{s} are merged (and
#' sorted) with \code{object$lambda}, and the model is refit before predictions
#' are made. In this case, it is required to supply the original data \code{x=}
#' and \code{y=} as additional named arguments to \code{predict()} or
#' \code{coef()}.  The workhorse \code{predict.glmnet()} needs to \code{update}
#' the model, and so needs the data used to create it. The same is true of
#' \code{weights}, \code{offset}, \code{penalty.factor}, \code{lower.limits},
#' \code{upper.limits} if these were used in the original call. Failure to do
#' so will result in an error.
#' @param newoffset If an offset is used in the fit, then one must be supplied
#' for making predictions (except for \code{type="coefficients"} or
#' \code{type="nonzero"})
#' @param \dots This is the mechanism for passing arguments like \code{x=} when
#' \code{exact=TRUE}; see\code{exact} argument.
#' @return The object returned depends on type.
#' @author Jerome Friedman, Trevor Hastie and Rob Tibshirani\cr Maintainer:
#' Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glmnet}, and \code{print}, and \code{coef} methods, and
#' \code{cv.glmnet}.
#' @references Friedman, J., Hastie, T. and Tibshirani, R. (2008)
#' \emph{Regularization Paths for Generalized Linear Models via Coordinate
#' Descent (2010), Journal of Statistical Software, Vol. 33(1), 1-22},
#' \doi{10.18637/jss.v033.i01}.\cr
#' Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2011)
#' \emph{Regularization Paths for Cox's Proportional
#' Hazards Model via Coordinate Descent, Journal of Statistical Software, Vol.
#' 39(5), 1-13},
#' \doi{10.18637/jss.v039.i05}.\cr
#' Glmnet webpage with four vignettes, \url{https://glmnet.stanford.edu}.
#' @keywords models regression
#'
#' @examples
#' x=matrix(rnorm(100*20),100,20)
#' y=rnorm(100)
#' g2=sample(1:2,100,replace=TRUE)
#' g4=sample(1:4,100,replace=TRUE)
#' fit1=glmnet(x,y)
#' predict(fit1,newx=x[1:5,],s=c(0.01,0.005))
#' predict(fit1,type="coef")
#' fit2=glmnet(x,g2,family="binomial")
#' predict(fit2,type="response",newx=x[2:5,])
#' predict(fit2,type="nonzero")
#' fit3=glmnet(x,g4,family="multinomial")
#' predict(fit3,newx=x[1:3,],type="response",s=0.01)
#' @method predict glmnet
#' @export
#' @export predict.glmnet
predict.grpnet=function(object,newx,s=NULL,type=c("link","response","coefficients"),newoffset,...){
 type=match.arg(type)
  if(missing(newx)){
    if(!match(type,c("coefficients"),FALSE))stop("You need to supply a value for 'newx'")
  }
 if(type=="response")stop("type response not yet implemented")
 state = object$state
 a0=t(as.matrix(state$intercepts))
 rownames(a0)="(Intercept)"
 betas = as(t(state$betas),"CsparseMatrix")
 np = dim(betas);nlam=np[2];p=np[1]
 dimnames(betas)=list(paste0("V",seq(p)),paste0("s",seq(nlam)))
  nbeta=methods::rbind2(a0,betas)
  if(!is.null(s)){
    vnames=dimnames(nbeta)[[1]]
    dimnames(nbeta)=list(NULL,NULL)
    lambda=state$lmdas
    lamlist=lambda.interp(lambda,s)
    nbeta=nbeta[,lamlist$left,drop=FALSE]%*%Diagonal(x=lamlist$frac) +nbeta[,lamlist$right,drop=FALSE]%*%Diagonal(x=1-lamlist$frac)
    namess=names(s)
    if(is.null(namess))namess=paste0("s",seq(0,length=length(s)))
    dimnames(nbeta)=list(vnames,namess)
  }
  if(type=="coefficients")return(nbeta)
   ###Check on newx
 if(inherits(newx, "sparseMatrix"))newx=as(newx,"dMatrix")
 dx=dim(newx)
 if(is.null(dx))newx=matrix(newx,1,byrow=TRUE)
 if(ncol(newx) != p)stop(paste0("The number of variables in newx must be ",p))
  nfit=as.matrix(cbind2(1,newx)%*%nbeta)
  ##  if(object$offset){
  ##   if(missing(newoffset))stop("No newoffset provided for prediction, yet offset used in fit of glmnet",call.=FALSE)
  ##   if(is.matrix(newoffset)&&inherits(object,"lognet")&&dim(newoffset)[[2]]==2)newoffset=newoffset[,2]
  ##   nfit=nfit+array(newoffset,dim=dim(nfit))
  ## }
nfit
  }


lambda.interp=function(lambda,s){
### lambda is the index sequence that is produced by the model
### s is the new vector at which evaluations are required.
### the value is a vector of left and right indices, and a vector of fractions.
### the new values are interpolated bewteen the two using the fraction
### Note: lambda decreases. you take:
### sfrac*left+(1-sfrac*right)

  if(length(lambda)==1){# degenerate case of only one lambda
    nums=length(s)
    left=rep(1,nums)
    right=left
    sfrac=rep(1,nums)
  }
  else{
      ## s[s > max(lambda)] = max(lambda)
      ## s[s < min(lambda)] = min(lambda)
      k=length(lambda)
      sfrac <- (lambda[1]-s)/(lambda[1] - lambda[k])
      lambda <- (lambda[1] - lambda)/(lambda[1] - lambda[k])
      sfrac[sfrac < min(lambda)] <- min(lambda)
      sfrac[sfrac > max(lambda)] <- max(lambda)
      coord <- approx(lambda, seq(lambda), sfrac)$y
      left <- floor(coord)
      right <- ceiling(coord)
      sfrac=(sfrac-lambda[right])/(lambda[left] - lambda[right])
      sfrac[left==right]=1
      sfrac[abs(lambda[left]-lambda[right])<.Machine$double.eps]=1

    }
list(left=left,right=right,frac=sfrac)
}

#' Extract coefficients from a glmnet object
#'
#' @method coef glmnet
#' @rdname predict.glmnet
#' @export
#' @export coef.glmnet
coef.grpnet=function(object,s=NULL,...)
  predict(object,s=s,type="coefficients",...)






