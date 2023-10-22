import { Children } from "react";

export default function SideBar({title, subtitle, description, children}){
    return <section className="totalSection">
        <div className="sideBar">
            <h3>{title}</h3>
            <h4>{subtitle}</h4>
            <div className="bar"></div>
            <p>{description}</p>
        </div>
        {Children.map(children,child=><div className="fc">{child}</div>)}
    </section>
}