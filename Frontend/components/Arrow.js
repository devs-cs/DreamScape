import { ArrowDownIcon } from "evergreen-ui";

export default function Arrow(){
    return <div className="arrow" id="arrow">
        <div className="arrowStem"></div>
        <ArrowDownIcon size={100} className="arrowIcon"></ArrowDownIcon>
    </div>
}