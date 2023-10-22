//progress is a percent, like .50
export default  function Loading({progress,text}){
    return <div className="loadingContainer">
        <div className="loadingBarContainer">
            <div className="loadProgress" style={{width: `${Math.round(progress*100)}%`}}></div>
            </div>
            {text&&<div className="loadingText">{text}</div>}
        </div>
}