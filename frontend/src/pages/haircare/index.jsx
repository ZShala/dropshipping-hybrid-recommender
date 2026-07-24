import React from "react";
import CategoryProducts from '../../components/category-products';
import SEO from '../../components/seo';

const Haircare = () => {
    return (
        <>
            <SEO 
                title="Hair Care Products | Professional Hair Solutions"
                description="Shop professional haircare products from top brands. Find shampoos, conditioners, treatments and styling products with Express delivery."
            />
            <div>
                <CategoryProducts categoryType="haircare" />
            </div>
        </>
    )
}

export default Haircare;